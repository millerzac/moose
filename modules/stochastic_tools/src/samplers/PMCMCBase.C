//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "PMCMCBase.h"
#include "AdaptiveMonteCarloUtils.h"
#include "Uniform.h"
#include "Normal.h"
#include "DelimitedFileReader.h"

registerMooseObject("StochasticToolsApp", PMCMCBase);

InputParameters
PMCMCBase::validParams()
{
  InputParameters params = Sampler::validParams();
  params.addClassDescription("Parallel Markov chain Monte Carlo base.");
  params.addRequiredParam<std::vector<DistributionName>>(
      "prior_distributions", "The prior distributions of the parameters to be calibrated.");
  params.addParam<DistributionName>(
      "prior_variance", "The prior distribution of the variance parameter to be calibrated.");
  params.addParam<std::vector<std::string>>(
      "variance_names", "Names for variance parameters (enables multi-variance mode).");
  params.addRequiredParam<unsigned int>(
      "num_parallel_proposals",
      "Number of proposals to make and corresponding subApps executed in "
      "parallel.");
  params.addRequiredParam<FileName>("file_name", "Name of the CSV file with configuration values.");
  params.addParam<std::string>(
      "file_column_name", "Name of column in CSV file to use, by default first column is used.");
  params.addParam<unsigned int>(
      "num_columns", "Number of columns to be used in the CSV file with the configuration values.");
  params.addParam<std::vector<Real>>("lower_bound", "Lower bounds for making the next proposal.");
  params.addParam<std::vector<Real>>("upper_bound", "Upper bounds for making the next proposal.");
  params.addRequiredParam<std::vector<Real>>("initial_values",
                                             "The starting values of the inputs to be calibrated.");
  params.addParam<unsigned int>(
      "num_random_seeds",
      100000,
      "Initialize a certain number of random seeds. Change from the default only if you have to.");
  return params;
}

PMCMCBase::PMCMCBase(const InputParameters & parameters)
  : Sampler(parameters),
    TransientInterface(this),
    _num_parallel_proposals(getParam<unsigned int>("num_parallel_proposals")),
    _multi_variance_mode(false),
    _lower_bound(isParamValid("lower_bound") ? &getParam<std::vector<Real>>("lower_bound")
                                             : nullptr),
    _upper_bound(isParamValid("upper_bound") ? &getParam<std::vector<Real>>("upper_bound")
                                             : nullptr),
    _check_step(0),
    _initial_values(getParam<std::vector<Real>>("initial_values")),
    _num_random_seeds(getParam<unsigned int>("num_random_seeds"))
{
  // Filling the `priors` vector with the user-provided distributions.
  for (const DistributionName & name :
       getParam<std::vector<DistributionName>>("prior_distributions"))
    _priors.push_back(&getDistributionByName(name));

  // MULTI-VARIANCE DETECTION: Only enable if user explicitly provides variance_names
  if (isParamValid("variance_names"))
  {
    const auto & var_names = getParam<std::vector<std::string>>("variance_names");
    if (!var_names.empty())
    {
      _multi_variance_mode = true;
      _variance_names = var_names;

      // Check if user wants MCMC variance estimation
      if (isParamValid("prior_variance"))
      {
        // MCMC variance estimation mode
        std::vector<DistributionName> var_priors =
            getParam<std::vector<DistributionName>>("prior_variance");

        if (var_priors.size() != var_names.size())
          mooseError("Number of prior_variance entries (",
                     var_priors.size(),
                     ") does not match variance_names (",
                     var_names.size(),
                     ")");

        _var_priors.clear();
        for (const auto & prior_name : var_priors)
          _var_priors.push_back(&getDistributionByName(prior_name));
      }
      else
      {
        // Multi-variable fixed noise mode
        _var_priors.clear();
      }

      // Setup multi-variable data structures
      _new_var_samples_vec.assign(_num_parallel_proposals,
                                  std::vector<Real>(var_names.size(), 0.0));
      _new_var_samples.assign(_num_parallel_proposals, 0.0);
      _var_prior = nullptr;
    }
  }
  else
  {
    // Legacy single-variance setup
    _multi_variance_mode = false;
    if (isParamValid("prior_variance"))
    {
      std::vector<DistributionName> var_priors =
          getParam<std::vector<DistributionName>>("prior_variance");
      _var_prior = &getDistributionByName(var_priors[0]);
    }
    else
    {
      _var_prior = nullptr;
    }
    _new_var_samples.assign(_num_parallel_proposals, 0.0);
    _variance_names.clear();
    _var_priors.clear();
  }

  // Read the experimental configurations from a csv file
  MooseUtils::DelimitedFileReader reader(getParam<FileName>("file_name"));
  reader.read();
  _confg_values.resize(1);
  if (isParamValid("file_column_name"))
    _confg_values[0] = reader.getData(getParam<std::string>("file_column_name"));
  else if (isParamValid("num_columns"))
  {
    _confg_values.resize(getParam<unsigned int>("num_columns"));
    for (unsigned int i = 0; i < _confg_values.size(); ++i)
      _confg_values[i] = reader.getData(i);
  }
  else
    _confg_values[0] = reader.getData(0);

  // Set sampler dimensions
  dof_id_type total_rows = _num_parallel_proposals * _confg_values[0].size();
  setNumberOfRows(total_rows);
  setNumberOfCols(_priors.size() + _confg_values.size());

  // Resize internal vectors
  _new_samples.resize(_num_parallel_proposals, std::vector<Real>(_priors.size(), 0.0));
  _new_samples_confg.resize(total_rows,
                            std::vector<Real>(_priors.size() + _confg_values.size(), 0.0));
  _rnd_vec.resize(_num_parallel_proposals);

  setNumberOfRandomSeeds(_num_random_seeds);
  _check_step = 0;

  // Simplified bounds validation
  if (_lower_bound && !_upper_bound)
    paramError("upper_bound", "Upper bounds must be specified when lower bounds are provided.");
  if (!_lower_bound && _upper_bound)
    paramError("lower_bound", "Lower bounds must be specified when upper bounds are provided.");

  if (_lower_bound && _upper_bound)
  {
    if (_lower_bound->size() != _upper_bound->size())
      mooseError("Lower and upper bounds must have the same size.");

    // Calculate expected bounds sizes
    unsigned int params_only = _priors.size();
    unsigned int params_and_variance = _priors.size();

    if (_multi_variance_mode && !_var_priors.empty())
      params_and_variance += _var_priors.size();
    else if (_var_prior)
      params_and_variance += 1;

    // Allow flexible bounds specification
    if (_lower_bound->size() != params_only && _lower_bound->size() != params_and_variance)
    {
      mooseError("Bounds array size (",
                 _lower_bound->size(),
                 ") must be either:\n",
                 "  - ",
                 params_only,
                 " (bounds for parameters only), or\n",
                 "  - ",
                 params_and_variance,
                 " (bounds for all parameters and variances)\n",
                 "Bounds order when including variances: [param1, param2, ..., var1, var2, ...]");
    }

    // Validate bounds are sensible (lower < upper)
    validateBounds();
  }

  // Check priors and initial values
  if (_priors.size() != _initial_values.size())
    mooseError("The priors and initial values should be of the same size.");

  if (_multi_variance_mode && _variance_names.empty())
    mooseError("Multi-variance mode enabled but no variance_names provided. "
               "Either provide variance_names or remove multi-variance configuration.");

  if (_multi_variance_mode && !_var_priors.empty() && _var_priors.size() != _variance_names.size())
    mooseError("Number of variance priors (",
               _var_priors.size(),
               ") must match variance names (",
               _variance_names.size(),
               ").");

  if (!_multi_variance_mode && isParamValid("previous_state_var_multi"))
    mooseWarning("previous_state_var_multi specified but multi-variance mode not enabled. "
                 "This parameter will be ignored.");
}

void
PMCMCBase::validateBounds() const
{
  if (!_lower_bound || !_upper_bound)
    return;

  // Check that all lower bounds are less than upper bounds
  for (unsigned int i = 0; i < _lower_bound->size(); ++i)
  {
    if ((*_lower_bound)[i] >= (*_upper_bound)[i])
    {
      // Provide context about which bound is problematic
      std::string param_type = "Parameter";
      unsigned int param_idx = i;

      if (i >= _priors.size())
      {
        param_type = "Variance";
        param_idx = i - _priors.size();

        if (_multi_variance_mode && param_idx < _variance_names.size())
          param_type += " '" + _variance_names[param_idx] + "'";
      }

      mooseError(param_type,
                 " ",
                 param_idx,
                 " has invalid bounds: [",
                 (*_lower_bound)[i],
                 ", ",
                 (*_upper_bound)[i],
                 "]. ",
                 "Lower bound must be less than upper bound.");
    }
  }
}

bool
PMCMCBase::checkBounds(const std::vector<Real> & values, unsigned int start_idx) const
{
  if (!_lower_bound || !_upper_bound)
    return true; // No bounds to check

  for (unsigned int i = 0; i < values.size(); ++i)
  {
    unsigned int bound_idx = start_idx + i;
    if (bound_idx >= _lower_bound->size())
      return true; // No bounds specified for this parameter

    if (values[i] < (*_lower_bound)[bound_idx] || values[i] > (*_upper_bound)[bound_idx])
      return false;
  }

  return true;
}

void
PMCMCBase::proposeSamples(const unsigned int seed_value)
{
  bool valid_sample;
  unsigned int attempts;
  const unsigned int max_attempts = 1000;

  for (unsigned int j = 0; j < _num_parallel_proposals; ++j)
  {
    valid_sample = false;
    attempts = 0;

    while (!valid_sample && attempts < max_attempts)
    {
      // Generate parameter proposals
      for (unsigned int i = 0; i < _priors.size(); ++i)
        _new_samples[j][i] = _priors[i]->quantile(getRand(seed_value));

      // Check parameter bounds
      valid_sample = checkBounds(_new_samples[j], 0);

      // Generate and check variance proposals if needed
      if (valid_sample)
      {
        if (_multi_variance_mode && !_var_priors.empty())
        {
          for (unsigned int v = 0; v < _var_priors.size(); ++v)
            _new_var_samples_vec[j][v] = _var_priors[v]->quantile(getRand(seed_value));

          valid_sample = checkBounds(_new_var_samples_vec[j], _priors.size());
        }
        else if (_var_prior)
        {
          _new_var_samples[j] = _var_prior->quantile(getRand(seed_value));
          valid_sample = checkBounds({_new_var_samples[j]}, _priors.size());
        }
      }

      attempts++;
    }

    if (!valid_sample)
      mooseWarning("Failed to generate valid sample within bounds after ",
                   max_attempts,
                   " attempts for walker ",
                   j,
                   ". Consider adjusting bounds or distributions.");
  }
}

void
PMCMCBase::sampleSetUp(const SampleMode /*mode*/)
{
  if (_t_step < 1 || _check_step == _t_step)
    return;
  _check_step = _t_step;

  const unsigned int seed_value = _t_step > 0 ? (_t_step - 1) : 0;

  // Filling the new_samples vector of vectors with new proposal samples
  proposeSamples(seed_value);

  // Draw random numbers to facilitate decision making later on
  for (unsigned int walker = 0; walker < _num_parallel_proposals; ++walker)
    _rnd_vec[walker] = getRand(seed_value);
}

void
PMCMCBase::randomIndex(const unsigned int & upper_bound,
                       const unsigned int & exclude,
                       const unsigned int & seed,
                       unsigned int & req_index)
{
  req_index = exclude;
  while (req_index == exclude)
    req_index = getRandl(seed, 0, upper_bound);
}

void
PMCMCBase::randomIndexPair(const unsigned int & upper_bound,
                           const unsigned int & exclude,
                           const unsigned int & seed,
                           unsigned int & req_index1,
                           unsigned int & req_index2)
{
  randomIndex(upper_bound, exclude, seed, req_index1);
  req_index2 = req_index1;
  while (req_index1 == req_index2)
    randomIndex(upper_bound, exclude, seed, req_index2);
}

void
PMCMCBase::combineWithExperimentalConfig()
{
  unsigned int index1;
  int index2 = -1;
  std::vector<Real> tmp;
  for (unsigned int i = 0; i < _num_parallel_proposals * _confg_values[0].size(); ++i)
  {
    index1 = i % _num_parallel_proposals;
    if (index1 == 0)
      ++index2;
    tmp = _new_samples[index1];
    for (unsigned int j = 0; j < _confg_values.size(); ++j)
      tmp.push_back(_confg_values[j][index2]);
    _new_samples_confg[i] = tmp;
  }
}

const std::vector<Real> &
PMCMCBase::getRandomNumbers() const
{
  return _rnd_vec;
}

const std::vector<Real> &
PMCMCBase::getVarSamples() const
{
  return _new_var_samples;
}

const std::vector<const Distribution *>
PMCMCBase::getPriors() const
{
  return _priors;
}

const Distribution *
PMCMCBase::getVarPrior() const
{
  return _var_prior;
}

Real
PMCMCBase::computeSample(dof_id_type row_index, dof_id_type col_index)
{
  if (_t_step < 1)
    for (unsigned int i = 0; i < _num_parallel_proposals; ++i)
      _new_samples[i] = _initial_values;

  // Combine the proposed samples with experimental configurations
  combineWithExperimentalConfig();

  // ADD SAFETY CHECKS
  if (row_index >= _new_samples_confg.size())
  {
    mooseError("Row index ",
               row_index,
               " out of bounds for _new_samples_confg (size: ",
               _new_samples_confg.size(),
               ")");
  }
  if (col_index >= _new_samples_confg[row_index].size())
  {
    mooseError("Col index ",
               col_index,
               " out of bounds for _new_samples_confg[",
               row_index,
               "] (size: ",
               _new_samples_confg[row_index].size(),
               ")");
  }

  return _new_samples_confg[row_index][col_index];
}
