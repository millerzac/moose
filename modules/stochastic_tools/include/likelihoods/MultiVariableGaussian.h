//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "MultiVariableLikelihoodFunctionBase.h"
#include "ReporterInterface.h"

/**
 * Multi-variable Gaussian likelihood function for MCMC with per-variable noise support
 */
class MultiVariableGaussian : public MultiVariableLikelihoodFunctionBase, public ReporterInterface
{
public:
  static InputParameters validParams();

  MultiVariableGaussian(const InputParameters & parameters);

  // Primary interface for MCMC - expects sample-major layout
  virtual Real function(const std::vector<Real> & flat_model_data) const override;

  // Required by PMCMCDecision for multi-variance mode
  virtual Real function(const std::vector<Real> & x,
                        const std::vector<Real> & noise_vec) const override;

protected:
  // Core evaluation methods
  Real evaluateWithFixedNoise(const std::vector<Real> & flat_model_data) const;
  Real evaluateWithReporterNoise(const std::vector<Real> & flat_model_data) const;

  // Noise reporter initialization
  void initializeNoiseReporters();

  // Experimental data access
  std::vector<Real> getExpVecForRow(unsigned int row) const;

  // Static likelihood computation utilities
  static Real computeLikelihood(const std::vector<Real> & exp,
                                const std::vector<Real> & model,
                                const std::vector<Real> & noise_vec,
                                const bool log_likelihood);

  static Real computeLikelihood(const std::vector<Real> & exp,
                                const std::vector<Real> & model,
                                const Real noise,
                                const bool log_likelihood);

  // Single-config interface for PMCMCDecision
  virtual Real function(const std::vector<Real> & model_data,
                        const std::vector<Real> & noise_vec,
                        int config_id) const override;

private:
  // Configuration
  const bool _log_likelihood;
  bool _use_fixed_noise;
  bool _multi_noise;
  bool _shared_noise_mode = false;

  // Fixed noise configuration
  std::vector<Real> _fixed_noise;

  // Reporter-based noise configuration
  std::vector<ReporterName> _noise_names;
  const Real * _noise_scalar_cache;
  std::vector<const std::vector<Real> *> _noise_vector_cache;

  // Initialization tracking
  bool _reporters_initialized;
  const Real * _noise_direct_fallback;

  // Experimental data storage
  std::vector<std::vector<Real>> _exp_columns;

  // Workspace vectors - allocated once, reused for all evaluations
  mutable std::vector<Real> _workspace_model;
  mutable std::vector<Real> _workspace_noise;

  // Cache experimental data for O(1) access instead of repeated getExpVecForRow calls
  std::vector<std::vector<Real>> _cached_exp_data;

  // Initialization flag for thread safety
  mutable bool _workspace_initialized = false;
};