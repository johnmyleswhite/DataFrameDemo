load("DataFrame/data.jl")
load("DataFrame/formula.jl")
load("Demo/lm.jl")

df = csvDataFrame("Demo/toy_example.csv")

with(df, :A)
df[1, 1]
df[1, "A"]

with(df, :(A + C))

#with(df, :(mean(A))) # Is broken

:(A ~ C)

#model = Formula(:(A ~ C)) # Was broken

model = Formula(:(A ~ B + C))

mf = model_frame(model, df)

mm = model_matrix(mf)

lm_fit = lm(:(A ~ B + C), df)
print(lm_fit)
