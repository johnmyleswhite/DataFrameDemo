load("lib/data.jl")
load("lib/formula.jl")
load("demo/lm.jl")

df = csvDataFrame("demo/toy_example.csv")

model = Formula(:(A ~ B + C))

mf = model_frame(model, df)

mm = model_matrix(mf)

lm_fit = lm(:(A ~ B + C), df)
print(lm_fit)
