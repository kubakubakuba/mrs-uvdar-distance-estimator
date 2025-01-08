set datafile separator ','

set terminal tikz size 12cm,8cm
set output 'fit.tex'

# Define functions
f1(x) = c1 + a1 / x**2
f2(x) = c2 + a2 / x**3
f3(x) = c3 + a3 / x**4
f4(x) = a4 * exp(b4 * x)

# Improved initial guesses
a1 = 1000; c1 = 100
a2 = 1000; c2 = 100
a3 = 1000; c3 = 100
a4 = 1000; b4 = -1

# Fit the data
fit f1(x) 'data.csv' using 1:2 via a1, c1
fit f2(x) 'data.csv' using 1:2 via a2, c2
fit f3(x) 'data.csv' using 1:2 via a3, c3
fit f4(x) 'data.csv' using 1:2 via a4, b4

set xlabel 'Distance (meters)'
set ylabel 'Average number of events'

# Define custom colors for the plots
set style line 1 lc rgb '#1f77b4' lw 2 # Blue for f1
set style line 2 lc rgb '#ff7f0e' lw 2 # Orange for f2
set style line 3 lc rgb '#2ca02c' lw 2 # Green for f3
set style line 4 lc rgb '#d62728' lw 2 # Red for f4
set style line 5 lc rgb '#9467bd' pt 7 ps 1.5 # Purple for data points

# Plot with better colors and legend formatting
set samples 100
set xrange [1:5]
set yrange [0:1200]
plot 'data.csv' using 1:2 with points linestyle 5 title 'Data Points', \
     f1(x) with lines linestyle 1 title sprintf("Fit: $y = %.2f \\times 10^{%d} / x^2 + %.2f \\times 10^{%d}$", a1/10**int(log10(a1)), int(log10(a1)), c1/10**int(log10(c1)), int(log10(c1))), \
     f2(x) with lines linestyle 2 title sprintf("Fit: $y = %.2f \\times 10^{%d} / x^3 + %.2f \\times 10^{%d}$", a2/10**int(log10(a2)), int(log10(a2)), c2/10**int(log10(c2)), int(log10(c2))), \
     f3(x) with lines linestyle 3 title sprintf("Fit: $y = %.2f \\times 10^{%d} / x^4 + %.2f \\times 10^{%d}$", a3/10**int(log10(a3)), int(log10(a3)), c3/10**int(log10(c3)), int(log10(c3))), \
     f4(x) with lines linestyle 4 title sprintf("Fit: $y = %.2f \\times 10^{%d} \\cdot e^{%.2f \\cdot x}$", a4/10**int(log10(a4)), int(log10(a4)), b4)
