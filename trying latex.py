import matplotlib.pyplot as plt

# Set the STIX font to resemble LaTeX style
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# Now create your plots as usual
fig, ax = plt.subplots()
ax.set_title(r'$\mathrm{Equation\,like\,this:}\ x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$')
ax.set_xlabel(r'$x$ axis label (mm)')
ax.set_ylabel(r'$y$ axis label (mm)')

plt.show()
