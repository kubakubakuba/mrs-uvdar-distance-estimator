set datafile separator ','

set terminal tikz size 12cm,8cm
set output 'fit.tex'

f1(x) = c1 + a1 / x**2
f2(x) = c2 + a2 / x**3
f3(x) = c3 + a3 / x**4
f4(x) = a4 * exp(b4 * x)

a1 = 1
c1 = 1
a2 = 1
c2 = 1
a3 = 1
c3 = 1
a4 = 1
b4 = -0.1

fit f1(x) 'data.csv' using 1:2 via a1, c1
fit f2(x) 'data.csv' using 1:2 via a2, c2
fit f3(x) 'data.csv' using 1:2 via a3, c3
fit f4(x) 'data.csv' using 1:2 via a4, b4

set samples 100
set xrange [1:5]
set yrange [0:1200]

plot 'data.csv' using 1:2 with points pointtype 7 title 'Data Points', \
     f1(x) with lines linewidth 2 title sprintf("Fit: $x^{-2}$ (a1 = %.2f)", a1), \
     f2(x) with lines linewidth 2 title sprintf("Fit: $x^{-3}$ (a2 = %.2f)", a2), \
     f3(x) with lines linewidth 2 title sprintf("Fit: $x^{-4}$ (a3 = %.2f)", a3), \
     f4(x) with lines linewidth 2 title sprintf("Fit: $a e^{b x}$ (a4 = %.2f, b4 = %.2f)", a4, b4)
