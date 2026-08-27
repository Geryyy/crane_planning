// The two things every file in this package writes when it refuses or reports.
//
// `crane_model::Status` is the frozen error type of
// `wiki/implementation/model_api_contract.md` 3 and the planner adds nothing to
// it. What the planner adds is a *habit*: a refusal is a code plus a sentence
// naming what could not be done, and a duration in a message is written the same
// way everywhere so two refusals out of two stages read as one voice. Both were
// a private copy in nine and two translation units respectively before they
// were one; the copies were identical, which is what made them worth removing.

#ifndef CRANE_PLANNING__STATUS_HPP_
#define CRANE_PLANNING__STATUS_HPP_

#include <string>
#include <utility>

#include "crane_model/model.hpp"

namespace crane_planning
{

/// A refusal carrying a code and the sentence that says what it refused.
[[nodiscard]] inline crane_model::Status failure(
  crane_model::ErrorCode code, std::string message)
{
  return crane_model::Status{code, std::move(message)};
}

/// A duration as it appears inside a message.
[[nodiscard]] inline std::string seconds(double value)
{
  return std::to_string(value) + " s";
}

}  // namespace crane_planning

#endif  // CRANE_PLANNING__STATUS_HPP_
