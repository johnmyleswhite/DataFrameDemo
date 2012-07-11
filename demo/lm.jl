type OLSResults
  call::Formula
  predictors::Matrix
  predictor_names::Vector
  responses::Matrix
  coefficients::Matrix
  std_errors::Matrix
  t_stats::Matrix
  p_values::Matrix
  predictions::Matrix
  residuals::Matrix
  log_likelihood::Float
  r_squared::Float
end

function print(results::OLSResults)
  println()
  println()
  println("Call to lm(): $(results.call)")
  println()
  println("Fitted Model:")
  println()
  printf(" %16s  %8.s  %8.s  %8.s  %8.s\n", "Term", "Estimate", "Std. Error", "t", "p-Value")
  N = size(results.coefficients, 1)
  for i = 1:N
    printf(" %16.s  ", results.predictor_names[i])
    println(join(map(z -> sprintf("%5.7f", z), {results.coefficients[i, 1],
                  results.std_errors[i, 1],
                  results.t_stats[i, 1],
                  results.p_values[i, 1]}), "  "))
  end
  println()
  println()
end

# minimal first version: support y ~ x1 + x2 + log(x3)
function lm(ex::Expr, df::DataFrame)
  call = Formula(ex)
  mf = model_frame(call, df)
  mm = model_matrix(mf)
  x = mm.model
  y = mm.response
  coefficients = inv(x' * x) * x' * y
  OLSResults(call,
             x,
             mm.model_colnames,
             y,
             coefficients,
             coefficients,
             coefficients,
             coefficients,
             x * coefficients,
             y - x * coefficients,
             0.0,
             0.0)
end
