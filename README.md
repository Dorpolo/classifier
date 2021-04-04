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
       * In order to re-run the full proccess (chossing hyper params, trianing the model accordingly and genreate results), you can easily render the RMD file after doing the following actions:
           1. Installing Python's virual envoirment (see bellow)
           2. Uncommenting the 2 python lines in the relevant chunk
           3. Configuring required `iteration_range` (list object) hyper param param. 
       
       * Please consider that this process will take some time as the model will be trained.
           * For quick run - please choose small values in **iteration_range** class param, and the proccess will be run quickly

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
  * Rstudio
  * Python - The R project should include Python virtual envoirment inside the project directory (can be installed using the [following link](https://support.rstudio.com/hc/en-us/articles/360023654474-Installing-and-Configuring-Python-with-RStudio))

       1. installing virtualenv package
       ```
       pip install virtualenv
       ```
       
       2. cd'ing localy to the cloned repository 
             
       ```
       cd <project-dir>
       ```
       
       3. triggering new virtual env
       
       ```
       virtualenv python
       ```
       
       4. activating env
       
       ```
       source python/bin/activate
       ```
       
       5. installing project's reauired python's packages
       
       ```
       pip install pandas plotly catboost sklearn numpy
       ```
    
  

                 
