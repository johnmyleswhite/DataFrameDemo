load("lib/data.jl")
load("lib/formula.jl")
load("demo/lm.jl")

with(df, :A)
df[1, 1]
df[1, "A"]

with(df, :(A + C))

#with(df, :(mean(A))) # Is broken
