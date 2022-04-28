
Jan,

The code which you should run is in 'run.py'. It contains code for a qMC experiment as well a large PCE experiment. I have chosen five variables for this initial analysis. 

'sensitivity.py' contains my own functions for calculating the first and total order Sobol indices from the fitted polynomial. 

I have made some effort to refactor the code and annotate it better. It is still a bit soupy at points where I have to import data and construct dictionaries etc. I hope it is not too opaque. I have highlighted both places where it may be parallised if you think that is appropriate. 

Run time on my laptop (intel i3, single thread) stands at around 0.4s. 

If you have any questions you can let me know by email. 

J 