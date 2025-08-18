//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "MultiVariableGaussian.h"
#include "Normal.h"
#include "DelimitedFileReader.h"
#include "MooseError.h"

registerMooseObject("StochasticToolsApp", MultiVariableGaussian);

InputParameters
MultiVariableGaussian::validParams()
{
  InputParameters params = MultiVariableLikelihoodFunctionBase::validParams();
  params.addClassDescription(
      "Multi-variable Gaussian likelihood function for MCMC with per-variable noise support.");
  params.addParam<bool>("log_likelihood", true, "Compute log-likelihood or likelihood.");
  params.addParam<bool>("multi_variance", true, "Allows for unique variance for each variable.");
  params.addRequiredParam<std::vector<ReporterName>>(
      "noise", "Reporter(s) for noise: single reporter (shared) or one per variable.");
  params.addRequiredParam<FileName>("file_name", "CSV file with experimental values.");
  params.addParam<std::vector<std::string>>(
      "file_column_names", "Column names in CSV file (uses all columns if not specified).");
  params.addParam<std::vector<Real>>("fixed_noise",
                                     "Fixed noise values (alternative to reporters).");
  return params;
}

MultiVariableGaussian::MultiVariableGaussian(const InputParameters & parameters)
  : MultiVariableLikelihoodFunctionBase(parameters),
    ReporterInterface(this),
    _log_likelihood(getParam<bool>("log_likelihood")),
    _use_fixed_noise(false),
    _multi_noise(false),
    _noise_scalar_cache(nullptr),
    _reporters_initialized(false),
    _noise_direct_fallback(nullptr)
{
  // Read experimental data from CSV
  MooseUtils::DelimitedFileReader reader(getParam<FileName>("file_name"));
  reader.read();
  _exp_columns.clear();

  if (isParamValid("file_column_names"))
  {
    const auto & column_names = getParam<std::vector<std::string>>("file_column_names");
    for (const auto & name : column_names)
      _exp_columns.push_back(reader.getData(name));
  }
  else
  {
    const auto & col_names = reader.getNames();
    for (size_t i = 0; i < col_names.size(); ++i)
      _exp_columns.push_back(reader.getData(col_names[i]));
  }

  if (_exp_columns.empty())
    mooseError("No experimental data columns found in file.");

  // Configure noise handling
  if (isParamValid("fixed_noise"))
  {
    _fixed_noise = getParam<std::vector<Real>>("fixed_noise");
    _use_fixed_noise = true;

    bool multi_variance = getParam<bool>("multi_variance");
    const size_t expected_size = _exp_columns.size();

    if (!multi_variance && _fixed_noise.size() != 1)
      mooseError("multi_variance=false requires exactly one fixed_noise value.");
    else if (multi_variance && _fixed_noise.size() != expected_size)
      mooseError("multi_variance=true requires ",
                 expected_size,
                 " fixed_noise values (got ",
                 _fixed_noise.size(),
                 ").");

    _multi_noise = multi_variance;
  }
  else
  {
    // Reporter-based noise configuration
    _noise_names = getParam<std::vector<ReporterName>>("noise");
    bool multi_variance = getParam<bool>("multi_variance");
    const size_t expected_size = _exp_columns.size();

    if (!multi_variance)
    {
      if (_noise_names.size() != 1)
        mooseError("multi_variance=false requires exactly one noise reporter.");
      _multi_noise = false;

      // Attempt early initialization
      try
      {
        _noise_direct_fallback = &getReporterValue<Real>(_noise_names[0]);
      }
      catch (...)
      {
        _noise_direct_fallback = nullptr;
      }
    }
    else
    {
      if (_noise_names.size() == 1)
      {
        // Shared noise across all variables
        _multi_noise = false;
        _shared_noise_mode = true;

        try
        {
          _noise_direct_fallback = &getReporterValue<Real>(_noise_names[0]);
        }
        catch (...)
        {
          _noise_direct_fallback = nullptr;
        }
      }
      else if (_noise_names.size() == expected_size)
      {
        // Per-variable noise
        _multi_noise = true;
        _shared_noise_mode = false;
      }
      else
      {
        mooseError("multi_variance=true requires either 1 reporter (shared) or ",
                   expected_size,
                   " reporters (per-variable), got ",
                   _noise_names.size(),
                   ".");
      }
    }
  }
  // Validate experimental data consistency
  if (_exp_columns.size() > 1)
  {
    const size_t expected_rows = _exp_columns[0].size();
    for (size_t i = 1; i < _exp_columns.size(); ++i)
    {
      if (_exp_columns[i].size() != expected_rows)
        mooseError("Experimental data column ",
                   i,
                   " has ",
                   _exp_columns[i].size(),
                   " rows, expected ",
                   expected_rows,
                   " to match first column");
    }
  }

  if (_exp_columns.size() < 1)
    mooseError("At least one experimental variable column is required.");

  // Validate noise configuration early
  if (_multi_noise && _noise_names.size() != _exp_columns.size())
    mooseError("In multi-noise mode, number of noise reporters (",
               _noise_names.size(),
               ") must match number of experimental variables (",
               _exp_columns.size(),
               ").");

  // OPTIMIZATION: Cache experimental data once for O(1) access
  const size_t num_exp_samples = _exp_columns[0].size();
  _cached_exp_data.reserve(num_exp_samples);

  for (size_t row = 0; row < num_exp_samples; ++row)
  {
    std::vector<Real> exp_row;
    exp_row.reserve(_exp_columns.size());
    for (size_t var = 0; var < _exp_columns.size(); ++var)
    {
      if (row >= _exp_columns[var].size())
        mooseError("Row ",
                   row,
                   " out of range for variable ",
                   var,
                   " (size: ",
                   _exp_columns[var].size(),
                   ").");
      exp_row.push_back(_exp_columns[var][row]);
    }
    _cached_exp_data.emplace_back(std::move(exp_row));
  }

  // Pre-allocate workspace vectors
  const size_t max_variables = _exp_columns.size();
  _workspace_model.resize(max_variables);
  _workspace_noise.resize(max_variables);
  _workspace_initialized = true;
}

void
MultiVariableGaussian::initializeNoiseReporters()
{
  if (_reporters_initialized || _use_fixed_noise)
    return;

  if (!_multi_noise)
  {
    // Single/shared noise reporter
    if (_noise_direct_fallback)
      _noise_scalar_cache = _noise_direct_fallback;
    else
    {
      try
      {
        _noise_scalar_cache =
            &const_cast<MultiVariableGaussian *>(this)->getReporterValueByName<Real>(
                _noise_names[0]);
      }
      catch (const std::exception & e)
      {
        mooseError("Failed to initialize noise reporter '", _noise_names[0], "': ", e.what());
      }
    }
  }
  else
  {
    // Per-variable noise reporters
    try
    {
      _noise_vector_cache.clear();
      for (auto & name : _noise_names)
      {
        _noise_vector_cache.push_back(
            &const_cast<MultiVariableGaussian *>(this)->getReporterValueByName<std::vector<Real>>(
                name, REPORTER_MODE_DISTRIBUTED));
      }
    }
    catch (const std::exception & e)
    {
      mooseError("Failed to initialize per-variable noise reporters: ", e.what());
    }
  }

  _reporters_initialized = true;
}

std::vector<Real>
MultiVariableGaussian::getExpVecForRow(unsigned int row) const
{
  if (_exp_columns.empty())
    mooseError("No experimental data available.");

  if (row >= _exp_columns[0].size())
    mooseError("Row ", row, " out of range (max: ", _exp_columns[0].size() - 1, ").");

  std::vector<Real> exp_vec;
  exp_vec.reserve(_exp_columns.size());

  for (size_t var = 0; var < _exp_columns.size(); ++var)
  {
    if (row >= _exp_columns[var].size())
      mooseError("Row ",
                 row,
                 " out of range for variable ",
                 var,
                 " (size: ",
                 _exp_columns[var].size(),
                 ").");
    exp_vec.push_back(_exp_columns[var][row]);
  }

  return exp_vec;
}

Real
MultiVariableGaussian::computeLikelihood(const std::vector<Real> & exp,
                                         const std::vector<Real> & model,
                                         const std::vector<Real> & noise_vec,
                                         const bool log_likelihood)
{
  mooseAssert(exp.size() == model.size() && model.size() == noise_vec.size(),
              "Vector size mismatch in likelihood computation.");

  Real result = log_likelihood ? 0.0 : 1.0;

  for (size_t i = 0; i < exp.size(); ++i)
  {
    Real pdf = Normal::pdf(exp[i], model[i], noise_vec[i]);
    mooseAssert(pdf > 0.0, "Invalid PDF value for likelihood computation.");

    if (log_likelihood)
      result += std::log(pdf);
    else
      result *= pdf;
  }

  return result;
}

Real
MultiVariableGaussian::computeLikelihood(const std::vector<Real> & exp,
                                         const std::vector<Real> & model,
                                         const Real noise,
                                         const bool log_likelihood)
{
  mooseAssert(exp.size() == model.size(), "Vector size mismatch in likelihood computation.");

  Real result = log_likelihood ? 0.0 : 1.0;

  for (size_t i = 0; i < exp.size(); ++i)
  {
    Real pdf = Normal::pdf(exp[i], model[i], noise);
    mooseAssert(pdf > 0.0, "Invalid PDF value for likelihood computation.");

    if (log_likelihood)
      result += std::log(pdf);
    else
      result *= pdf;
  }

  return result;
}

Real
MultiVariableGaussian::function(const std::vector<Real> & flat_model_data) const
{
  // Primary MCMC interface - expects sample-major layout:
  // [sample0_var0, sample0_var1, ..., sample0_varN, sample1_var0, ...]

  if (_use_fixed_noise)
    return evaluateWithFixedNoise(flat_model_data);

  const_cast<MultiVariableGaussian *>(this)->initializeNoiseReporters();
  return evaluateWithReporterNoise(flat_model_data);
}

Real
MultiVariableGaussian::function(const std::vector<Real> & x,
                                const std::vector<Real> & noise_vec) const
{
  // Required by PMCMCDecision for multi-variance mode
  if (_exp_columns.empty())
    mooseError("No experimental data available.");

  std::vector<Real> exp_vec = getExpVecForRow(0);
  if (exp_vec.size() != x.size() || x.size() != noise_vec.size())
    mooseError("Vector size mismatch: exp(",
               exp_vec.size(),
               "), model(",
               x.size(),
               "), noise(",
               noise_vec.size(),
               ").");

  return computeLikelihood(exp_vec, x, noise_vec, _log_likelihood);
}

Real
MultiVariableGaussian::evaluateWithFixedNoise(const std::vector<Real> & flat_model_data) const
{
  const size_t num_variables = _exp_columns.size();
  const size_t flat_size = flat_model_data.size();
  const size_t num_exp_samples = _exp_columns[0].size();

  if (num_variables == 1)
  {
    // Single-variable case - use cached experimental data
    const std::vector<Real> & exp_vec = _cached_exp_data[0];
    return computeLikelihood(exp_vec, flat_model_data, _fixed_noise[0], _log_likelihood);
  }

  // Multi-variable case with sample-major layout
  if (flat_size % num_variables != 0)
    mooseError("Data size (", flat_size, ") not divisible by variables (", num_variables, ").");

  const size_t num_samples = flat_size / num_variables;
  if (num_samples > num_exp_samples)
    mooseError(
        "Model samples (", num_samples, ") exceed experimental data (", num_exp_samples, ").");

  Real total_likelihood = 0.0;

  for (size_t sample = 0; sample < num_samples; ++sample)
  {
    // Direct reference to cached experimental data - no allocation
    const std::vector<Real> & exp_vec = _cached_exp_data[sample];

    // Reuse workspace vector - no allocation
    for (size_t var = 0; var < num_variables; ++var)
      _workspace_model[var] = flat_model_data[sample * num_variables + var];

    if (_multi_noise)
      total_likelihood +=
          computeLikelihood(exp_vec, _workspace_model, _fixed_noise, _log_likelihood);
    else
      total_likelihood +=
          computeLikelihood(exp_vec, _workspace_model, _fixed_noise[0], _log_likelihood);
  }

  return total_likelihood;
}

Real
MultiVariableGaussian::evaluateWithReporterNoise(const std::vector<Real> & flat_model_data) const
{
  const size_t num_variables = _exp_columns.size();
  const size_t flat_size = flat_model_data.size();
  const size_t num_exp_samples = _exp_columns[0].size();

  // Single-variable fallback (unchanged for compatibility)
  if (num_variables == 1)
  {
    const std::vector<Real> & exp_vec = _cached_exp_data[0]; // Use cached data
    return computeLikelihood(exp_vec, flat_model_data, *_noise_scalar_cache, _log_likelihood);
  }

  // Multi-variable case
  if (flat_size % num_variables != 0)
    mooseError("Data size (", flat_size, ") not divisible by variables (", num_variables, ").");

  const size_t num_samples = flat_size / num_variables;
  if (num_samples > num_exp_samples)
    mooseError(
        "Model samples (", num_samples, ") exceed experimental data (", num_exp_samples, ").");

  Real total_likelihood = 0.0;

  // ONE-TIME validation outside loop for per-variable noise mode
  if (_multi_noise && !_shared_noise_mode)
  {
    if (_noise_vector_cache.size() != num_variables)
      mooseError("Noise reporter count (",
                 _noise_vector_cache.size(),
                 ") != variables (",
                 num_variables,
                 ").");

    for (size_t var = 0; var < _noise_vector_cache.size(); ++var)
    {
      if (!_noise_vector_cache[var])
        mooseError("Null noise reporter for variable ", var);
      if (_noise_vector_cache[var]->size() != num_samples)
        mooseError("Noise reporter for variable ",
                   var,
                   " has ",
                   _noise_vector_cache[var]->size(),
                   " samples, expected ",
                   num_samples);
    }
  }

  // Main evaluation loop with zero allocations
  for (size_t sample = 0; sample < num_samples; ++sample)
  {
    // Direct reference to cached experimental data - no allocation or copy
    const std::vector<Real> & exp_vec = _cached_exp_data[sample];

    // Reuse workspace vector - no allocation
    for (size_t var = 0; var < num_variables; ++var)
      _workspace_model[var] = flat_model_data[sample * num_variables + var];

    if (_multi_noise && !_shared_noise_mode)
    {
      // Reuse workspace vector - no allocation
      for (size_t var = 0; var < num_variables; ++var)
        _workspace_noise[var] = (*_noise_vector_cache[var])[sample];

      total_likelihood +=
          computeLikelihood(exp_vec, _workspace_model, _workspace_noise, _log_likelihood);
    }
    else
    {
      // Shared noise across all variables
      total_likelihood +=
          computeLikelihood(exp_vec, _workspace_model, *_noise_scalar_cache, _log_likelihood);
    }
  }

  return total_likelihood;
}

Real
MultiVariableGaussian::function(const std::vector<Real> & model_data,
                                const std::vector<Real> & noise_vec,
                                int config_id) const
{
  if (config_id < 0 || static_cast<size_t>(config_id) >= _exp_columns[0].size())
    mooseError("Config ID ", config_id, " out of range (0-", _exp_columns[0].size() - 1, ").");

  std::vector<Real> exp_vec = getExpVecForRow(config_id);
  return computeLikelihood(exp_vec, model_data, noise_vec, _log_likelihood);
}