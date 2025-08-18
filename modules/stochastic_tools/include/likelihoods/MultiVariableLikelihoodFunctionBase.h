//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "LikelihoodFunctionBase.h"

/**
 * Base class for multi-variable likelihood functions used in MCMC simulations.
 * Provides interfaces for flat data evaluation and multi-variable noise handling.
 */
class MultiVariableLikelihoodFunctionBase : public LikelihoodFunctionBase
{
public:
  static InputParameters validParams() { return LikelihoodFunctionBase::validParams(); }

  MultiVariableLikelihoodFunctionBase(const InputParameters & parameters)
    : LikelihoodFunctionBase(parameters)
  {
  }

  /**
   * Primary interface for MCMC evaluation with flattened multi-variable data.
   * Expected layout: sample-major [sample0_var0, sample0_var1, ..., sample1_var0, ...]
   * @param flat_model_data Flattened vector containing all samples and variables
   * @return Combined likelihood or log-likelihood value
   */
  virtual Real function(const std::vector<Real> & flat_model_data) const = 0;
};
