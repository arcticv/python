# Description

Course notebook for Python and VBA


# Project Python

## Forecasting
1. visualize
1. stationarize
1. plot charts
1. make model (arima/ML)
1. make predictions

## KNN K Nearest Neighbors Model
1. Read the file
1. Impute missing values 
1. Drop categorical variables (non numerical), drop item ID column, and drop output column
1. Standarize the inputs (scaling the features)
1. Create training set
1. Analyze error (RMSE) on y axis across different k values in x axis - use Elbow Method by choosing minimum error but not too high k (next marginal benefit is not worth it)
1. Elbow method short cut is using GridSearch algo

## Python and PowerBI
1. Data pulled in comes through <b>dataset</b>['Date', 'ColumnA']
1. Pandas is implicitly called. You can access it using pandas.DataFrame()
1. Do not import entire sklearn, it'll be very slow
1. Try not to create dataframes within because csv files are generated on the fly that may increase size

# Release Notes
Date | Action
------------ | -------------
March 26, 2020 | * Opened repository <br> * Added matplotlib example <br> * Added MSSQL server example
March 27, 2020 | * Added readme.md
April 23, 2020 | * Update for PowerBI



## Getting Started
Test
```
Give examples
```

### Installing
A step by step series of examples that tell you how to get a development env running
Say what the step will be
```
Give the example
```
And repeat
```
until finished
```
End with an example of getting some data out of the system or using it for a little demo


## Testing Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used

## Acknowledgments
* Links to cool people
  * Readme Sample [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2#file-readme-template-md)
* Inspiration
* etc

## Authors

* **Vincent** - *Initial work* - [arcticv](https://github.com/arcticv/)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.
Please read [CONTRIBUTING.md](https://gist.github.com/) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
