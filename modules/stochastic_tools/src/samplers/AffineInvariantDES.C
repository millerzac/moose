//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "AffineInvariantDES.h"
#include "Normal.h"
#include "Uniform.h"

registerMooseObject("StochasticToolsApp", AffineInvariantDES);

/*
 Tuning options for the internal parameters
  1. Braak2006_static:
  - the gamma param is set to 2.38 / sqrt(2 * dim)
  - the b param is set to (scale) * 1e-6
*/

InputParameters
AffineInvariantDES::validParams()
{
  InputParameters params = PMCMCBase::validParams();
  params.addClassDescription("Perform Affine Invariant Ensemble MCMC with differential sampler.");
  params.addRequiredParam<ReporterName>(
      "previous_state", "Reporter value with the previous state of all the walkers.");
  params.addParam<ReporterName>(
      "previous_state_var",
      "Reporter value with the previous state of all the walkers for variance.");
  params.addParam<std::vector<ReporterName>>(
      "previous_state_var_multi",
      "Vector of previous variance reporter values (multi-variance mode, one per variable).");
  MooseEnum tuning_option("Braak2006_static", "Braak2006_static");
  params.addParam<MooseEnum>(
      "tuning_option", tuning_option, "The tuning option for internal parameters.");
  params.addParam<std::vector<Real>>("scales", "Scales for the parameters.");
  return params;
}

AffineInvariantDES::AffineInvariantDES(const InputParameters & parameters)
  : PMCMCBase(parameters),
    _previous_state(getReporterValue<std::vector<std::vector<Real>>>("previous_state")),
    _tuning_option(getParam<MooseEnum>("tuning_option"))
{
  // Configure variance state tracking based on sampler mode
  if (_multi_variance_mode)
  {
    if (!isParamValid("previous_state_var_multi"))
      paramError("previous_state_var_multi",
                 "Must specify previous_state_var_multi when variance_names is provided.");

    const auto & reporters = getParam<std::vector<ReporterName>>("previous_state_var_multi");

    _previous_state_var_multi.reserve(reporters.size());
    for (const auto & reporter_name : reporters)
      _previous_state_var_multi.push_back(
          &getReporterValueByName<std::vector<Real>>(reporter_name));
  }
  else
  {
    if (!isParamValid("previous_state_var"))
      paramError("previous_state_var",
                 "Must specify previous_state_var when variance_names is not provided.");
    _previous_state_var_single = &getReporterValue<std::vector<Real>>("previous_state_var");
  }

  // Validation checks
  if (_num_parallel_proposals < 5)
    paramError(
        "num_parallel_proposals",
        "At least five parallel proposals should be used for the Differential Evolution Sampler.");

  if (_num_parallel_proposals < _priors.size())
    mooseWarning(
        "It is recommended that the parallel proposals be greater than or equal to the "
        "inferred parameters. This will allow the sampler to not get stuck on a hyper-plane.");

  if (isParamValid("scales"))
  {
    _scales = getParam<std::vector<Real>>("scales");
    if (_scales.size() != _priors.size())
      paramError("scales",
                 "The number of scales provided should match the number of tunable params.");
  }
  else
  {
    _scales.assign(_priors.size(), 1.0);
  }
  // OPTIMIZATION: Pre-allocate workspace for all walkers to eliminate repeated allocations
  _proposal_workspace.resize(_num_parallel_proposals);
  for (auto & workspace : _proposal_workspace)
    workspace.resize(_priors.size());

  if (_multi_variance_mode && !_var_priors.empty())
  {
    _variance_workspace.resize(_num_parallel_proposals);
    for (auto & workspace : _variance_workspace)
      workspace.resize(_var_priors.size());
  }
}

void
AffineInvariantDES::computeDifferential(
    const Real & state1, const Real & state2, const Real & rnd, const Real & scale, Real & diff)
{
  Real gamma;
  Real b;
  tuneParams(gamma, b, scale);
  diff = gamma * (state1 - state2) + Normal::quantile(rnd, 0.0, b);
}

void
AffineInvariantDES::tuneParams(Real & gamma, Real & b, const Real & scale)
{
  if (_tuning_option == "Braak2006_static")
  {
    gamma = 2.38 / std::sqrt(2 * _priors.size());
    b = 1e-6 * scale;
  }
}

void
AffineInvariantDES::proposeSamples(const unsigned int seed_value)
{
  if (_multi_variance_mode)
  {
    // Multi-variance mode with workspace optimization
    for (unsigned int j = 0; j < _num_parallel_proposals; ++j)
    {
      bool valid_proposal = false;
      unsigned int attempts = 0;
      const unsigned int max_attempts = 1000;

      while (!valid_proposal && attempts < max_attempts)
      {
        unsigned int index_req1, index_req2;
        randomIndexPair(_num_parallel_proposals, j, seed_value, index_req1, index_req2);

        // Generate parameter proposals using pre-allocated workspace
        std::vector<Real> & proposed_params = _proposal_workspace[j];
        for (unsigned int i = 0; i < _priors.size(); ++i)
        {
          if (_t_step > decisionStep())
          {
            Real diff;
            computeDifferential(_previous_state[index_req1][i],
                                _previous_state[index_req2][i],
                                getRand(seed_value),
                                _scales[i],
                                diff);
            proposed_params[i] = _previous_state[j][i] + diff;
          }
          else
          {
            proposed_params[i] = _priors[i]->quantile(getRand(seed_value));
          }
        }

        // Check parameter bounds
        valid_proposal = checkBounds(proposed_params, 0);

        if (valid_proposal)
        {
          // Single copy instead of multiple reallocations
          _new_samples[j] = proposed_params;

          // Generate variance proposals using workspace
          if (!_var_priors.empty())
          {
            std::vector<Real> & proposed_variances = _variance_workspace[j];

            for (unsigned int v = 0; v < _var_priors.size(); ++v)
            {
              const auto & prev_var = (*_previous_state_var_multi[v]);

              if (_t_step <= decisionStep() || prev_var.empty())
              {
                proposed_variances[v] = _var_priors[v]->quantile(getRand(seed_value));
              }
              else
              {
                Real diff;
                computeDifferential(
                    prev_var[index_req1], prev_var[index_req2], getRand(seed_value), 1.0, diff);
                proposed_variances[v] = prev_var[j] + diff;
              }
            }

            // Check variance bounds
            valid_proposal = checkBounds(proposed_variances, _priors.size());

            if (valid_proposal)
              _new_var_samples_vec[j] = proposed_variances; // Single copy
          }
        }

        attempts++;
      }

      if (!valid_proposal)
      {
        mooseWarning("Walker ",
                     j,
                     " failed to generate valid proposal after ",
                     max_attempts,
                     " attempts. Using fallback.");
        if (j < _previous_state.size())
          _new_samples[j] = _previous_state[j];
        else
          _new_samples[j] = _initial_values;
      }
    }
  }
  else
  {
    unsigned int j = 0;
    bool indicator;
    unsigned int index_req1, index_req2;
    Real diff;
    while (j < _num_parallel_proposals)
    {
      indicator = false;
      randomIndexPair(_num_parallel_proposals, j, seed_value, index_req1, index_req2);

      for (unsigned int i = 0; i < _priors.size(); ++i)
      {
        computeDifferential(_previous_state[index_req1][i],
                            _previous_state[index_req2][i],
                            getRand(seed_value),
                            _scales[i],
                            diff);
        _new_samples[j][i] = (_t_step > decisionStep()) ? (_previous_state[j][i] + diff)
                                                        : _priors[i]->quantile(getRand(seed_value));
        if (_lower_bound)
          indicator =
              (_new_samples[j][i] < (*_lower_bound)[i] || _new_samples[j][i] > (*_upper_bound)[i])
                  ? true
                  : indicator;
      }
      if (_var_prior)
      {
        computeDifferential((*_previous_state_var_single)[index_req1],
                            (*_previous_state_var_single)[index_req2],
                            getRand(seed_value),
                            1.0,
                            diff);
        _new_var_samples[j] = (_t_step > decisionStep())
                                  ? ((*_previous_state_var_single)[j] + diff)
                                  : _var_prior->quantile(getRand(seed_value));
        if (_new_var_samples[j] < 0.0)
          indicator = true;
      }

      if (!indicator)
        ++j;
    }
  }
}
