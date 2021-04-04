# Home Assignment

## Project Structure

  * configs.py - holds feature metadata and loading project dataset.
  * classifier.py - main python's class, holds all project python's functions within a class called ModelExplained.
  * data_wrangler.r - holds all R functions that have been used for data exploration proccess
  * output.RMD - pre generated project's final output as .RMD file.
  
  * **output.html** - MAIN PROJECT OUTPUT, generated project's final output as .html file.
  
  * data
      * dataset.csv - project dataset
      * plots
         * .csv files that produced during the python's ModelExplained class run. Served as datasets for project plots.

---

 ## Project Presentation
  
  * The presented HTML holds the final trained model, after picking hyper parameters from the following ranges:
      * Learning Rate: [0.16, 0.17, 0.18, 0.19]
      * Depth: [7, 8, 9]
      * Trees (iterations): [500, 5000]

  * The output presents the best performed model, based on the above params, including evluation metrics comparison.

  * PLEASE NOTE: 
       * In order to re-run the model, you can easily render the RMD file after uncommenting the 2 python lines in the relevant chunk (and choose required trees learning hyper params range). This process will take time as model will be trained.
       * For quick run - please choose small values in **iteration_range** class param, and the proccess will be run very quickly

```{python}
from classifier import ModelExplained

try:
   # instance = ModelExplained(iteration_range=[500, 5000])
   # instance.main()
   print('Uncomment commands in order to trigger the model training proccess')
except Exception as e:
   print(e)
```

--- 

## Prequisits 

  * Python - The project inclding python virtual-env with all the relevant pip installs.
  * Rstudio.

                 
