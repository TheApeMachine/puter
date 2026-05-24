package hlo

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
)

const activeInferenceEpsilon = 1e-12

func RenderBeliefUpdate(
	moduleName string,
	elementFormat dtype.DType,
	count int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	vectorLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	entryLayout := fmt.Sprintf("%s,%s->%s", vectorLiteral, vectorLiteral, vectorLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] add(lhs, rhs)
}

ENTRY main {
  likelihood = %s parameter(0)
  prior = %s parameter(1)
  product = %s multiply(likelihood, prior)
  zero = %s[] constant(0)
  total = %s[] reduce(product, zero), dimensions={0}, to_apply=%%add
  inv = %s[] divide(%s[] constant(1), total)
  inv_b = %s broadcast(inv), dimensions={0}
  ROOT result = %s multiply(product, inv_b)
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		vectorLiteral, vectorLiteral, vectorLiteral,
		elementType, elementType, elementType, elementType, vectorLiteral, vectorLiteral), nil
}

func RenderPrecisionWeight(
	moduleName string,
	elementFormat dtype.DType,
	count int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	vectorLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	entryLayout := fmt.Sprintf("%s,%s->%s", vectorLiteral, vectorLiteral, vectorLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  errors = %s parameter(0)
  precision = %s parameter(1)
  ROOT result = %s multiply(errors, precision)
}
`, moduleName, entryLayout, vectorLiteral, vectorLiteral, vectorLiteral), nil
}

func RenderFreeEnergy(
	moduleName string,
	elementFormat dtype.DType,
	count int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	vectorLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	scalarLiteral := fmt.Sprintf("%s[]", elementType)
	entryLayout := fmt.Sprintf("%s,%s,%s->%s", vectorLiteral, vectorLiteral, vectorLiteral, scalarLiteral)
	epsilon := activeInferenceEpsilon

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] add(lhs, rhs)
}

ENTRY main {
  likelihood = %s parameter(0)
  posterior = %s parameter(1)
  prior = %s parameter(2)
  eps = %s[] constant(%g)
  like_clamped = %s maximum(likelihood, eps)
  post_clamped = %s maximum(posterior, eps)
  prior_clamped = %s maximum(prior, eps)
  log_like = %s log(like_clamped)
  log_post = %s log(post_clamped)
  log_prior = %s log(prior_clamped)
  cross_term = %s multiply(posterior, log_like)
  kl_post = %s subtract(log_post, log_prior)
  kl_term = %s multiply(posterior, kl_post)
  cross_neg = %s negate(cross_term)
  zero = %s[] constant(0)
  cross_sum = %s[] reduce(cross_neg, zero), dimensions={0}, to_apply=%%add
  kl_sum = %s[] reduce(kl_term, zero), dimensions={0}, to_apply=%%add
  ROOT result = %s add(cross_sum, kl_sum)
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		vectorLiteral, vectorLiteral, vectorLiteral,
		elementType, epsilon,
		vectorLiteral, vectorLiteral, vectorLiteral,
		vectorLiteral, vectorLiteral, vectorLiteral,
		vectorLiteral, vectorLiteral, vectorLiteral,
		vectorLiteral, elementType, scalarLiteral, scalarLiteral, scalarLiteral), nil
}

func RenderExpectedFreeEnergy(
	moduleName string,
	elementFormat dtype.DType,
	obsCount, stateCount int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	obsLiteral := fmt.Sprintf("%s[%d]{0}", elementType, obsCount)
	stateLiteral := fmt.Sprintf("%s[%d]{0}", elementType, stateCount)
	scalarLiteral := fmt.Sprintf("%s[]", elementType)
	entryLayout := fmt.Sprintf("%s,%s,%s->%s", obsLiteral, obsLiteral, stateLiteral, scalarLiteral)
	epsilon := activeInferenceEpsilon

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] add(lhs, rhs)
}

ENTRY main {
  predicted_obs = %s parameter(0)
  preferred_obs = %s parameter(1)
  predicted_state = %s parameter(2)
  eps = %s[] constant(%g)
  pred_clamped = %s maximum(predicted_obs, eps)
  pref_clamped = %s maximum(preferred_obs, eps)
  state_clamped = %s maximum(predicted_state, eps)
  log_pred = %s log(pred_clamped)
  log_pref = %s log(pref_clamped)
  log_state = %s log(state_clamped)
  pragmatic_diff = %s subtract(log_pred, log_pref)
  pragmatic = %s multiply(predicted_obs, pragmatic_diff)
  epistemic_neg = %s multiply(predicted_state, log_state)
  epistemic = %s negate(epistemic_neg)
  zero = %s[] constant(0)
  pragmatic_sum = %s[] reduce(pragmatic, zero), dimensions={0}, to_apply=%%add
  epistemic_sum = %s[] reduce(epistemic, zero), dimensions={0}, to_apply=%%add
  ROOT result = %s add(pragmatic_sum, epistemic_sum)
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		obsLiteral, obsLiteral, stateLiteral,
		elementType, epsilon,
		obsLiteral, obsLiteral, stateLiteral,
		obsLiteral, obsLiteral, stateLiteral,
		obsLiteral, obsLiteral, stateLiteral, stateLiteral,
		elementType, elementType, elementType, scalarLiteral), nil
}
