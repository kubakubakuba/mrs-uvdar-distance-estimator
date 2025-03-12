set terminal tikz size 12cm,8cm
set output 'mapping_gnuplot.tex'

#set title 'Projections with f = 1'
set xlabel '$\theta$ (radians)'
set ylabel 'r'

set xrange [0:pi]
set yrange [0:3]

set grid

f = 1

r1(theta) = f * tan(theta)
r2(theta) = 2 * f * tan(theta / 2)
r3(theta) = f * theta
r4(theta) = 2 * f * sin(theta / 2)
r5(theta) = f * sin(theta)

set style line 1 lc rgb "#0000FF" lw 2  # Blue
set style line 2 lc rgb "#FF0000" lw 2  # Red
set style line 3 lc rgb "#00FF00" lw 2  # Green
set style line 4 lc rgb "#FFA500" lw 2  # Orange
set style line 5 lc rgb "#800080" lw 2  # Purple

unset key
#plot r1(x) title 'Rectilinear (r = $f$ tan $\theta$)' with lines lw 2, \
#     r2(x) title 'Stereographic (r = 2$f$ tan($\theta$/2))' with lines lw 2, \
#     r3(x) title 'Equidistant (r = f $\theta$)' with lines lw 2, \
#     r4(x) title 'Equisolid angle (r = 2$f$ sin($\theta$/2))' with lines lw 2, \
#     r5(x) title 'Orthographic (r = $f$ sin $\theta$)' with lines lw 2

plot r1(x) ls 1, \
     r2(x) ls 2, \
     r3(x) ls 3, \
     r4(x) ls 4, \
     r5(x) ls 5

set key outside right