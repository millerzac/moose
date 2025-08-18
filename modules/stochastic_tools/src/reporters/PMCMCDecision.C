//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "PMCMCDecision.h"
#include "Sampler.h"
#include "DenseMatrix.h"
#include "Uniform.h"

registerMooseObject("StochasticToolsApp", PMCMCDecision);

InputParameters
PMCMCDecision::validParams()
{
  InputParameters params = GeneralReporter::validParams();
  params += LikelihoodInterface::validParams();
  params.addClassDescription("Generic reporter which decides whether or not to accept a proposed "
                             "sample in parallel Markov chain Monte Carlo type of algorithms.");
  params.addRequiredParam<ReporterName>("output_value",
                                        "Value of the model output from the SubApp.");
  params.addParam<ReporterValueName>(
      "outputs_required",
      "outputs_required",
      "Modified value of the model output from this reporter class.");
  params.addParam<ReporterValueName>("inputs", "inputs", "Uncertain inputs to the model.");
  params.addParam<ReporterValueName>("tpm", "tpm", "The transition probability matrix.");
  params.addParam<ReporterValueName>("variance", "variance", "Model variance term.");
  params.addParam<ReporterValueName>(
      "noise", "noise", "Model noise term to pass to Likelihoods object.");
  params.addRequiredParam<SamplerName>("sampler", "The sampler object.");
  params.addRequiredParam<std::vector<UserObjectName>>("likelihoods", "Names of likelihoods.");
  return params;
}

PMCMCDecision::PMCMCDecision(const InputParameters & parameters)
  : GeneralReporter(parameters),
    LikelihoodInterface(parameters),
    _output_value(getReporterValue<std::vector<Real>>("output_value", REPORTER_MODE_DISTRIBUTED)),
    _outputs_required(declareValue<std::vector<Real>>("outputs_required")),
    _inputs(declareValue<std::vector<std::vector<Real>>>("inputs")),
    _tpm(declareValue<std::vector<Real>>("tpm")),
    _variance(declareValue<std::vector<Real>>("variance")),
    _sampler(getSampler("sampler")),
    _pmcmc(dynamic_cast<const PMCMCBase *>(&_sampler)),
    _rnd_vec(_pmcmc->getRandomNumbers()),
    _new_var_samples(_pmcmc->getVarSamples()),
    _priors(_pmcmc->getPriors()),
    _var_prior(_pmcmc->getVarPrior()),
    _local_comm(_sampler.getLocalComm()),
    _check_step(std::numeric_limits<int>::max())
{
  if (!_pmcmc)
    paramError("sampler", "Sampler must be of type PMCMCBase.");

  // Configure noise handling based on sampler mode
  _multi_noise_mode = _pmcmc->isMultiVarianceMode();

  if (_multi_noise_mode)
  {
    // Multi-variance mode: declare per-variable noise/variance reporters
    _variance_names = _pmcmc->getVarianceNames();

    if (_variance_names.empty())
      mooseError("Multi-variance mode requires variance names from sampler.");

    _variance_vec_reporters.reserve(_variance_names.size());
    _noise_vec_reporters.reserve(_variance_names.size());

    for (const auto & var_name : _variance_names)
    {
      std::string variance_reporter = "variance_" + var_name;
      std::string noise_reporter = "noise_" + var_name;

      auto & var_ref =
          declareValueByName<std::vector<Real>>(variance_reporter, REPORTER_MODE_DISTRIBUTED);
      auto & noise_ref =
          declareValueByName<std::vector<Real>>(noise_reporter, REPORTER_MODE_DISTRIBUTED);

      _variance_vec_reporters.push_back(&var_ref);
      _noise_vec_reporters.push_back(&noise_ref);
    }
  }
  else
  {
    // Legacy single-variance mode
    _noise = &declareValue<Real>("noise");
  }

  // Initialize likelihood functions
  for (const auto & name : getParam<std::vector<UserObjectName>>("likelihoods"))
    _likelihoods.push_back(getLikelihoodFunctionByName(name));

  // Check whether the selected sampler is an MCMC sampler or not
  if (!_pmcmc)
    paramError("sampler", "The selected sampler is not of type MCMC.");

  // Fetching the sampler characteristics
  _props = _pmcmc->getNumParallelProposals();
  _num_confg_values = _pmcmc->getNumberOfConfigValues();
  _num_confg_params = _pmcmc->getNumberOfConfigParams();

  // Resizing the data arrays to transmit to the output file
  _inputs.resize(_props);
  const dof_id_type input_size = _sampler.getNumberOfCols() - _num_confg_params;
  for (unsigned int i = 0; i < _props; ++i)
    _inputs[i].resize(input_size);
  _outputs_required.resize(_sampler.getNumberOfRows());
  _tpm.resize(_props);
  _variance.resize(_props);

  // OPTIMIZATION: Pre-allocate workspace for evidence computation
  const size_t max_variables = _multi_noise_mode ? _variance_names.size() : 1;
  _evidence_workspace_current.resize(max_variables);
  _evidence_workspace_previous.resize(max_variables);
  _evidence_workspace_noise_current.resize(max_variables);
  _evidence_workspace_noise_previous.resize(max_variables);
}

void
PMCMCDecision::computeEvidence(std::vector<Real> & evidence, const DenseMatrix<Real> & input_matrix)
{
  for (unsigned int i = 0; i < evidence.size(); ++i)
  {
    evidence[i] = 0.0;
    for (unsigned int j = 0; j < _priors.size(); ++j)
    {
      evidence[i] += std::log(_priors[j]->pdf(input_matrix(i, j))) -
                     std::log(_priors[j]->pdf(_data_prev(i, j)));
    }

    if (_multi_noise_mode)
    {
      // Multi-variable mode with workspace optimization
      const dof_id_type num_variables = _variance_names.size();

      // Variance prior evidence (compute once per walker)
      if (!_pmcmc->getVarPriors().empty() && !_var_prev_vec.empty())
      {
        const auto & walker_vars = _variance_vec[i];
        const auto & prev_walker_vars = _var_prev_vec[i];

        for (unsigned int v = 0; v < walker_vars.size(); ++v)
        {
          evidence[i] += std::log(_pmcmc->getVarPriors()[v]->pdf(walker_vars[v])) -
                         std::log(_pmcmc->getVarPriors()[v]->pdf(prev_walker_vars[v]));
        }

        // Create noise vectors using workspace - no allocation
        for (unsigned int v = 0; v < walker_vars.size(); ++v)
        {
          _evidence_workspace_noise_current[v] = std::sqrt(walker_vars[v]);
          _evidence_workspace_noise_previous[v] = std::sqrt(prev_walker_vars[v]);
        }

        // Process each configuration separately using workspace
        for (unsigned int config = 0; config < _num_confg_values; ++config)
        {
          // Extract variables for this specific config using workspace - no allocation
          for (unsigned int var = 0; var < num_variables; ++var)
          {
            const dof_id_type sample_idx = config * _props + i;
            const dof_id_type output_idx = sample_idx * num_variables + var;
            _evidence_workspace_current[var] = _outputs_required[output_idx];
            _evidence_workspace_previous[var] = _outputs_prev[output_idx];
          }

          // Call likelihood for this configuration only
          for (const auto & likelihood : _likelihoods)
          {
            evidence[i] += likelihood->function(
                _evidence_workspace_current, _evidence_workspace_noise_current, config);
            evidence[i] -= likelihood->function(
                _evidence_workspace_previous, _evidence_workspace_noise_previous, config);
          }
        }
      }
      else
      {
        // Fixed noise mode for multi-variable using workspace
        for (unsigned int config = 0; config < _num_confg_values; ++config)
        {
          // Use workspace - no allocation
          for (unsigned int var = 0; var < num_variables; ++var)
          {
            const dof_id_type sample_idx = config * _props + i;
            const dof_id_type output_idx = sample_idx * num_variables + var;
            _evidence_workspace_current[var] = _outputs_required[output_idx];
            _evidence_workspace_previous[var] = _outputs_prev[output_idx];
          }

          for (const auto & likelihood : _likelihoods)
          {
            evidence[i] += likelihood->function(_evidence_workspace_current) -
                           likelihood->function(_evidence_workspace_previous);
          }
        }
      }
    }
    else
    {
      // Legacy single-variable mode (unchanged for compatibility)
      std::vector<Real> out1(_num_confg_values);
      std::vector<Real> out2(_num_confg_values);

      for (unsigned int j = 0; j < _num_confg_values; ++j)
      {
        out1[j] = _outputs_required[j * _props + i];
        out2[j] = _outputs_prev[j * _props + i];
      }

      if (_var_prior)
      {
        evidence[i] += std::log(_var_prior->pdf(_new_var_samples[i])) -
                       std::log(_var_prior->pdf(_var_prev[i]));

        *_noise = std::sqrt(_new_var_samples[i]);
        for (const auto & likelihood : _likelihoods)
          evidence[i] += likelihood->function(out1);

        *_noise = std::sqrt(_var_prev[i]);
        for (const auto & likelihood : _likelihoods)
          evidence[i] -= likelihood->function(out2);
      }
      else
      {
        for (const auto & likelihood : _likelihoods)
          evidence[i] += likelihood->function(out1) - likelihood->function(out2);
      }
    }
  }
}

void
PMCMCDecision::computeTransitionVector(std::vector<Real> & tv,
                                       const std::vector<Real> & /*evidence*/)
{
  tv.assign(_props, 1.0);
}

void
PMCMCDecision::nextSamples(std::vector<Real> & req_inputs,
                           DenseMatrix<Real> & input_matrix,
                           const std::vector<Real> & tv,
                           const unsigned int & parallel_index)
{
  if (tv[parallel_index] >= _rnd_vec[parallel_index])
  {
    for (unsigned int k = 0; k < _sampler.getNumberOfCols() - _num_confg_params; ++k)
      req_inputs[k] = input_matrix(parallel_index, k);
    _variance[parallel_index] = _new_var_samples[parallel_index];
  }
  else
  {
    for (unsigned int k = 0; k < _sampler.getNumberOfCols() - _num_confg_params; ++k)
    {
      req_inputs[k] = _data_prev(parallel_index, k);
      input_matrix(parallel_index, k) = _data_prev(parallel_index, k);
    }
    if (_var_prior)
      _variance[parallel_index] = _var_prev[parallel_index];

    // Revert outputs - ORIGINAL LOGIC FOR BOTH MODES
    if (_multi_noise_mode)
    {
      const dof_id_type num_variables = _variance_names.size();
      for (unsigned int k = 0; k < _num_confg_values; ++k)
      {
        for (unsigned int var = 0; var < num_variables; ++var)
        {
          const dof_id_type sample_idx = k * _props + parallel_index;
          const dof_id_type output_idx = sample_idx * num_variables + var;
          _outputs_required[output_idx] = _outputs_prev[output_idx];
        }
      }
    }
    else
    {
      for (unsigned int k = 0; k < _num_confg_values; ++k)
        _outputs_required[k * _props + parallel_index] = _outputs_prev[k * _props + parallel_index];
    }
  }
}

void
PMCMCDecision::execute()
{
  if (_sampler.getNumberOfLocalRows() == 0 || _check_step == _t_step)
  {
    _check_step = _t_step;
    return;
  }

  // Gather inputs and outputs from the sampler and subApps
  DenseMatrix<Real> data_in(_sampler.getNumberOfRows(), _sampler.getNumberOfCols());
  for (dof_id_type ss = _sampler.getLocalRowBegin(); ss < _sampler.getLocalRowEnd(); ++ss)
  {
    const auto data = _sampler.getNextLocalRow();
    for (unsigned int j = 0; j < _sampler.getNumberOfCols(); ++j)
      data_in(ss, j) = data[j];
  }
  _local_comm.sum(data_in.get_values());
  _outputs_required = _output_value;
  _local_comm.allgather(_outputs_required);
  // Get variance data for multi-variance mode
  if (_multi_noise_mode)
    _variance_vec = _pmcmc->getVarSamplesVec();

  // Compute the evidence and transition vectors
  std::vector<Real> evidence(_props);
  if (_t_step > _pmcmc->decisionStep())
  {
    // Initialize _var_prev_vec on first evidence call
    if (_multi_noise_mode && _var_prev_vec.empty())
      _var_prev_vec = _variance_vec;

    computeEvidence(evidence, data_in);
    computeTransitionVector(_tpm, evidence);
  }
  else
  {
    _tpm.assign(_props, 1.0);
  }

  // Accept/reject the proposed samples and assign the correct outputs
  std::vector<Real> req_inputs(_sampler.getNumberOfCols() - _num_confg_params);
  for (unsigned int i = 0; i < _props; ++i)
  {
    nextSamples(req_inputs, data_in, _tpm, i);
    _inputs[i] = req_inputs;
  }

  // Update variance/noise reporters for multi-variance mode
  if (_multi_noise_mode)
  {
    const bool has_var_priors = !_pmcmc->getVarPriors().empty();

    for (unsigned int v = 0; v < _variance_names.size(); ++v)
    {
      std::vector<Real> variance_vals(_props);
      std::vector<Real> noise_vals(_props);

      if (has_var_priors && !_variance_vec.empty())
      {
        for (unsigned int w = 0; w < _props; ++w)
        {
          const Real var_val = _variance_vec[w][v];
          variance_vals[w] = var_val;
          noise_vals[w] = std::sqrt(var_val);
        }
      }
      else
      {
        std::fill(variance_vals.begin(), variance_vals.end(), 1.0);
        std::fill(noise_vals.begin(), noise_vals.end(), 1.0);
      }

      *_variance_vec_reporters[v] = variance_vals;
      *_noise_vec_reporters[v] = noise_vals;
    }
  }
  // Compute the next seeds to facilitate proposals (not always required)
  nextSeeds();

  // Store data from previous step
  _data_prev = data_in;
  _outputs_prev = _outputs_required;
  if (!_multi_noise_mode)
    _var_prev = _variance;
  else if (!_var_prev_vec.empty())
    _var_prev_vec = _variance_vec;

  // Track the current step
  _check_step = _t_step;
}
