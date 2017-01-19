Spectro
=======
This program analyzes UV-VIS absorption spectra from aqueous surfactant suspended dispersions of carbon nanotubes.
It does so by fitting absorption profile models from the literature with a linear regression at each step of a non-linear regression fitting of the background (amorphous carbon and pi plasmon resonances) model.

This program is written using python 2.7 with a Qt4 Gui and has the following dependencies:

* [pythonxy](http://www.mirrorservice.org/sites/pythonxy.com/Python(x,y)-2.7.6.1.exe)
* [lmfit](https://pypi.python.org/packages/2.7/l/lmfit/lmfit-0.7.4.win32-py2.7.exe)
* xlsxwriter
* scipy

This is a screen shot of the program which has fit an absorption spectra of the provided (6,5)/(6,4) semiconducting enriched SWCNT sample:
![ScreenShot](http://imgur.com/ybhHaOf.jpg)


This program was written by Chase Brown and is under the GNU General Public License.  (A Copyleft License)

This program is free to use for any research, as long as this this thesis and github page are cited:

https://github.com/chaxor/Spectro

Brown, Chase. “CARBON NANOTUBES AND MAGNETIC PARTICLES AS PICKERING EMULSION STABALIZERS: 
PARTICLE CONTROL FOR PHASE SELECTIVE REACTIONS” Master’s thesis, University of Oklahoma, 2016.
