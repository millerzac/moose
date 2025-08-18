//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "MultiVariableReporter.h"

registerMooseObject("StochasticToolsApp", MultiVariableReporter);

InputParameters
MultiVariableReporter::validParams()
{
  InputParameters params = GeneralReporter::validParams();
  params.addClassDescription(
      "Aggregates multiple Reporter values into sample-major layout for multi-variable MCMC.");
  params.addRequiredParam<std::vector<ReporterName>>(
      "reporters", "List of Reporter values to aggregate (one per variable).");
  return params;
}

MultiVariableReporter::MultiVariableReporter(const InputParameters & parameters)
  : GeneralReporter(parameters), _cache_initialized(false)
{
  const auto & reporter_names = getParam<std::vector<ReporterName>>("reporters");

  if (reporter_names.empty())
    mooseError("The 'reporters' parameter must contain at least one reporter.");

  // Declare output vector with distributed mode for parallel efficiency
  _all_variables =
      &declareValueByName<std::vector<Real>>("all_variables", REPORTER_MODE_DISTRIBUTED);

  // Pre-allocate cache vectors
  const size_t num_reporters = reporter_names.size();
  _is_real_type_cache.resize(num_reporters);
  _real_reporters.resize(num_reporters);
  _int_reporters.resize(num_reporters);
}

void
MultiVariableReporter::initialize()
{
  if (_cache_initialized)
    return;

  const auto & reporter_names = getParam<std::vector<ReporterName>>("reporters");

  // Cache type information and reporter pointers once during initialization
  for (size_t i = 0; i < reporter_names.size(); ++i)
  {
    const auto & r_name = reporter_names[i];

    if (hasReporterValueByName<std::vector<Real>>(r_name))
    {
      _is_real_type_cache[i] = true;
      _real_reporters[i] = &getReporterValueByName<std::vector<Real>>(r_name);
      _int_reporters[i] = nullptr;
    }
    else if (hasReporterValueByName<std::vector<int>>(r_name))
    {
      _is_real_type_cache[i] = false;
      _real_reporters[i] = nullptr;
      _int_reporters[i] = &getReporterValueByName<std::vector<int>>(r_name);
    }
    else
    {
      mooseError("Reporter '",
                 r_name.getCombinedName(),
                 "' must be of type std::vector<Real> or std::vector<int>.");
    }
  }

  _cache_initialized = true;
}

std::vector<Real>
MultiVariableReporter::aggregateVariables()
{
  if (!_cache_initialized)
    mooseError("MultiVariableReporter cache not initialized. Call initialize() first.");

  const size_t num_variables = _is_real_type_cache.size();

  // Determine number of samples from first reporter
  const size_t num_samples =
      _is_real_type_cache[0] ? _real_reporters[0]->size() : _int_reporters[0]->size();

  // Validate all reporters have same number of samples
  for (size_t i = 1; i < num_variables; ++i)
  {
    const size_t size =
        _is_real_type_cache[i] ? _real_reporters[i]->size() : _int_reporters[i]->size();
    if (size != num_samples)
      mooseError("All reporters must have the same number of samples. Reporter ",
                 i,
                 " has ",
                 size,
                 " samples, expected ",
                 num_samples,
                 ".");
  }

  // Resize buffer only if needed (avoid reallocation)
  const size_t required_size = num_samples * num_variables;
  if (_output_buffer.size() != required_size)
    _output_buffer.resize(required_size);

  // Direct assignment with pointer arithmetic for maximum speed
  Real * data_ptr = _output_buffer.data();
  for (size_t sample = 0; sample < num_samples; ++sample)
  {
    for (size_t var = 0; var < num_variables; ++var)
    {
      *data_ptr++ = _is_real_type_cache[var] ? (*_real_reporters[var])[sample]
                                             : static_cast<Real>((*_int_reporters[var])[sample]);
    }
  }

  return std::move(_output_buffer); // Move semantics - no copy
}

void
MultiVariableReporter::execute()
{
  *_all_variables = aggregateVariables();
}