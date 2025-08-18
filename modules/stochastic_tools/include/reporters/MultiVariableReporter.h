//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GeneralReporter.h"

/**
 * Aggregates multiple Reporter values into a single flat vector with sample-major layout
 * for use with multi-variable likelihood functions in MCMC simulations.
 */
class MultiVariableReporter : public GeneralReporter
{
public:
  static InputParameters validParams();
  MultiVariableReporter(const InputParameters & parameters);

  virtual void initialize() override;
  virtual void execute() override;
  virtual void finalize() override {}

private:
  /// Aggregates all reporter values into sample-major flat vector
  std::vector<Real> aggregateVariables();

  /// Reference to the aggregated output vector
  std::vector<Real> * _all_variables;

  // PERFORMANCE OPTIMIZATION: Cache reporter pointers and type information
  /// Cached type information for each reporter (true = Real, false = int)
  std::vector<bool> _is_real_type_cache;

  /// Cached pointers to Real reporters
  std::vector<const std::vector<Real> *> _real_reporters;

  /// Cached pointers to int reporters
  std::vector<const std::vector<int> *> _int_reporters;

  /// Flag to track if cache has been initialized
  bool _cache_initialized;

  /// Pre-allocated output buffer to avoid repeated allocations
  mutable std::vector<Real> _output_buffer;
};