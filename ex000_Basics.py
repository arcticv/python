#Jupyter Notebook Keyboard Shortcuts
# Do not use ''' but use """ for block comment
"""
shift + tab				check function documentation
shift + enter 				run cell, select below.
ctrl + enter 				run cell
option/alt + enter 			run cell insert below.
A insert cell above.
B insert cell below.
C copy cell.
V paste cell.
D , D delete selected cell.
"""

# Basic Practice: http://codingbat.com/python
# More Mathematical (and Harder) Practice: https://projecteuler.net/archives
# List of Practice Problems: http://www.codeabbey.com/index/task_list
# A SubReddit Devoted to Daily Practice Problems: https://www.reddit.com/r/dailyprogrammer
# A very tricky website with very few hints and touch problems (Not for beginners but still interesting) http://www.pythonchallenge.com/



# Print Code
mystring = "Hello World"
print(mystring)
print(mystring[-2]) # l

#Slice
mystring = "0123456789"
print(mystring[0])  #index starts at 0
print(mystring[3:]) #prints 3 and onwards
print(mystring[:5]) #go up to but not including 5, returning 01234
print(mystring[3:6]) #prints and includes 3 up to but not include 6
print(mystring[::]) #beginning to end, step 1, same as print(mystring)
print(mystring[2:7:2]) #location 2 up to but excl 7, step 2
print(mystring[::-1]) #end to beginning

#Concatenate
mystring1 = 'Hello'
mystring2 = 'J' + mystring1[1:] #Jello    #strings are immutable
print(mystring2) #Jello
print(mystring2 + 'is Jello')
# for loop concatenate
for value,label in zip(result,labels):
    print(label + ' : ' +str(value))
    if result[1] <= 0.05:
        print('Reject null hypothesis')
    else:
        print('Fail to reject null hypothesis')


#Multiple strings
mystring = 'x' * 10
print(mystring) #xxxxxxxxxx

#Use String Functions
mystring = 'Hello World'
mystring = mystring.upper() #use tab to see functions
mystring.split() #['Hello', 'World']
mystring.split('l') #returns a list split by l

#Inserting a string
print('This is a string {}'.format('INSERTED'))
print('I like %s' %'apples')
print('Where did {0} {2} {1}'.format('He', 'Her', 'Meet')) # He Meet Her
print('The {b} {c} {d}'.format(d='dummy', c='called', b='boy')) #The boy called dummy

#Controlling outputs and FString
result = 100/777
print("The result was {r:1.5f}".format(r=result)) # R:whitespace.decimalsF The result was 0.129
name = 'Jose'
print(f'Hello, his name is {name}') #fstring string variable injection method

# Some other ways way number formatting
print('Mean Height: ', round(np.mean(height_surveys)/12,1), ' feet')
print('Standard Deviation of Height: ', round(np.var(height_surveys)**0.5 / 12, 1), ' feet')

#######################################################################################################################
#lists
my_list = [1,2,3]
my_list2 = ['STRING', 100, 200]
my_list3 = [0]*3 #[0, 0, 0]
my_list(1:)
len(my_list + my_list2)
new_list.append(4) #adds something to the end
new_list.pop(0) # removes position zero and returns it
new_list.sort() # a function that sorts in place and returns nothing
new_list2 = sorted(my_list)
type(new_list)

#nested list
my_list=[1,2,[a,b]]
my_list[2][1]

#generated random list 10000
# Calculate stats using single sample
sample_mean = np.mean(height_surveys[3])
sample_stdev = np.var(height_surveys[3])**0.5
# Calculate standard error = sample std dev / sqrt(N) where N = 10 in this example
std_error = sample_stdev/(height_surveys[3].shape[0])**0.5
# generate random list of 10000 normally distributed samples (sample mean normal distribution)
inferred_distribution = [sample_mean + np.random.normal()*\
                 std_error for i in range(10000)]


#dictionaries have use "string" key values, do not need exact index, can't sort (because it is a mapping and not a sequence), use for vlookup but can be fancy lookup of lists/nested lookup
my_dict = {'key1':'value1','key2':'value2'}
my_dict['key1']
prices_lookup = {'apple':2.99,'oranges':2,'milk':5}
prices_lookup['apple']
prices_lookup['apple'] = 5 #reassign and mutable
fancy_dict = {'k1':123,'k2':[0,1,2,3],'k3':{'insidekey':100}}
fancy_dict['k3']['insidekey']
len(prices_lookup)
type(prices_lookup)
prices_lookup.keys()			#returns (['key1', 'key2'])
prices_lookup.values()
prices_lookup.items()   #returns the package dict_items([('key1', 'value1'), ('key2', 'value2')]) and requires for loop to unpack it

#tricky mixed up
# Getting a little tricker
d = {'k1':[{'nest_key':['this is deep',['hello']]}]}
d['k1'][0]['nest_key'][1][0] #Grab hello

#tuples are similar to lists, but IMMUTABLE, so it's good for data integrity when passing around objects that can't change once defined (can't re-assign like lists can)
t = ('a','a','b')
x = ('a',[1,2],'c')
t.count('a')
t[0] = 'NEW' #will throw an error

#sets, sets do not allow duplicate items, they are unordered collections of UNIQUE elements
myset = set()
mylist = [1,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,3]
myset = set(mylist) #list injection into a set, returns no repeats values or uniquely 1, 2, 3
myset = set()
myset.add(1) #looks like dictionary with {} but no key value pairs, so not
myset.add(2)
myset.add(2) #won't throw error and won't repeat it

#######################################################################################################################
#boolean for comparison
True
False
1 > 2
1 == 2
1 != 2
b = None #to avoid object not defined yet


#comparisons and or not
'3' == 3 #returns false due to different types
'Bye' == 'bye' #returns false due to case sensitive
3.0 == 3 #returns true because both are numbers
3 != 4 #returns true
(1 < 2) and (2 > 3)
(1 < 2) or (2 > 3)
not (1 != 2)


#######################################################################################################################
#opening files
myfile = open('file.txt')
myfile = open('C:\\Users\\Username\\Folder\\test.txt')
pwd
contents = myfile.read()  #reading it again and the cursor is moved to the end
myfile.seek(0) #resets the cursor
myfile.readlines() #each line as a separate element in the list
myfile.close()

#more formal way of opening files, use shift tab
with open('myfile.txt', mode='r') as my_new_file:
	contents = my_new_file.read()
#open as r = readonly, w = overWrite/create, a = Append, r+ read and write, w+ overwrite existing and read
with open('myfile.txt', mode='a') as my_new_file:
	my_new_file.write('Four\n')



#*************************************************************************************************************

# control flows
hungry = True
location = 'Stadium'
if hungry:
	# do x
	print('x')
elif location=='Bank':
	# do y
	print('y')
else:
	# do z
	print('z')

#*************************************************************************************************************
	
# for loop mod
myList = [1,2,3]
for xItem in myList:
	print(xItem)
for xItem in myList:
	if xItem % 2 == 0:
		print(xItem)
	else:
		print(f'Odd Number: {xItem}')

# tuple unpacking will print the imitated structure
myList = [(1,2),(3,4),(5,6),(7,8)]
for (a,b) in myList
	print(a) # returns 1... 3, etc
	print(b) #returns 2... 4...
for a,b in myList   # no brackets works too 
	print(a) 
	print(b) 

# dictionary unpacking and dictionary is unordered
myDict = {'k1':1, 'k2':2, 'k3':3}
for xItem in myDict:
	print(xItem) # only returns the k1, k2...
for xItem in myDict.items():
	print(xItem) # would return the tuple ('k1', 1)
for xKey, xValue in myDict.items():
	print(xValue) # would return the content

# while statements
x = 0
while x < 5:
	print(f'X is {x}')
	x = x+1
# while statement with an else
x = 0
while x < 5:
	print(f'X is {x}')
	x = x+1
else:
	print('Code complete')

#loop management
break 		#breaks and stops out of closest loop, useful for while loop
pass		#go back to top of closest loop, skip the rest of the instructions
continue 	#do nothing

# pass: example in a 'for loop'
myList = [1,2,3]
for xItem in myList:
	#comment alone would not compile as compiler expects a tabbed action and comment is not an action
	pass
print('end of loop')

# continue: example in a 'for loop'
myString = 'hello world'
for xChar in myString:
	if xChar == 'o':
		continue 			#this would skip all the o but allow the loop to continue
	print(xChar)			#prints 'hell wrld' with each line break between the letters

# break: example in a for loop
myString = 'hello world'
for xChar in myString:
	if xChar == 'o':
		break 				#this would break out
	print(xChar)			#prints 'hell' with each line break between the letters

	
	
#*************************************************************************************************************

# range operator
for xItem in range(3,10):
	print(xItem) # prints 3 to 9 and not include 10
for xItem in range(0,10,2):
	print(xItem) # prints 2, 4, 6, 8, 10 (not incl 11)
myList = (range(0,10,2)) # generator a list using range

# query the list with for loop
index_count = 0
for letter in 'abcde':
	print(f'The index is: {index_count} and the letter is {letter})
	index_count += 1
	
# Access the sub components of a word or list
index_count = 0
word = 'hello'
for letter in word:
	print(word[index_count])
	index_count += 1
# Exact same but using enumerate
word = 'hello'
for item in enumerate(word):
	print(item) #returns tuples such as (0, 'h'), (1, 'e'), etc...
# Exact same but using enumerate and tuple unpacking
word = 'hello'
for index,letter in enumerate(word):
	print(index)
	print(letter) #returns tuple unpacked

	
#zip operator
my_list = [1, 2, 3, 4, 5, 6]
my_list2 = ['a', 'b', 'c', 'd', 'e']
for item in zip(my_list, my_list2):
	print(item) #returns tuples, (1, 'a') (2, 'b')
	print('\n') #the zip only goes to the shortest so ignores 6
my_list3 = list(zip(my_list,my_list2) #returns [(1,'a'), (2,'b'), (3,'c')]
#check if in list
2 in [1,2,3] 		#returns true
'a' in 'helloapple' #returns true
d = {'mykey':345} 	#returns true
'mykey' in d 		#returns true
345 in d.values 	#returns true
345 in d.keys 		#returns false
min(my_list) 		#returns 1
max(my_list)		#returns 6

#scramble the list
shuffle(my_list) 				#would reshuffle the list
from random import shuffle
random_list = shuffle(mylist)	#no error but doesn't return

#random integer
from random import randint
mynum = randint(0, 100) #returns a random integer

#input
results = input('What is your name')
results 		#returns always a string
type(results)	#returns string
float(results)	#cast to a number
int(results)	#cast to a number
results = int(input('Favorite Number')) 

#list append
mystring = 'hello'
mylist = []
for letter in mystring:
	mylist.append(letter) # ['h', 'e', 'l', 'l', 'o']

#list comprehension append 2, flattened for loop
mylist = [letter for letter in mystring]   # ['h', 'e', 'l', 'l', 'o']
mylist = [x for x in 'word']
mylist = [num for num in range(0,11)] #generate a series
mylist = [num**2 for num in range(0,11)] #perform the math operation
mylist = [x for x in range(0,11) if x%2 == 0] #only take the even numbers

st = 'Create a list of the first letters of every word in this string'
mylist = [x[0] for x in st.split()]    # ['C', 'a', 'l', 'o', 't', 'f', 'l', 'o', 'e', 'w', 'i', 't', 's']


#another example of list to list
c = [1, 2, 3, 4, 5]
f = [(( 9/5) * temp  + 32) for temp in c]    #load the list without for loop
#same as list to list but using append
listgrab = []
for x in c:
    listgrab.append((( 9/5) * x  + 32))		#load the list with for loop
	

#nested loop
mylist = []
for x in [2, 4, 6]:
	for y in [100, 200]:
		mylist.append(x*y)




#fizz buzz
mylist2 = []
for x in range(1, 101):
    if x % 15 == 0:
        mylist2.append('FizzBuzz')
    elif x % 5 == 0:
        mylist2.append('Buzz')
    elif x % 3 == 0:
        mylist2.append('Fizz')
    else:
        mylist2.append(x)
mylist2

#documentation
help(mylist.insert) # not help(mylist.insert())


#functions and functions that return
def some_function():
	'''
	DOCSTRING: Some documentation
	INPUT: expected input
	OUTPUT: expected output
	'''
	print('hello')

help(some_function) # would read the docstring
	
def some_function2(varX):
	print('x'+varX) # would throw error if no varX provided

def some_function_provide_nothing(name='NAME'):
	print('hello '+name) # has a default name

def simple_add(num1, num2):
	return num1 + num2

result = simple_add(1,2)
print(result) # expecting 3

def dog_check(mystring):
	if 'dog' in mystring.lower():
		return True
	else:
		return False

# better statement as it is already a boolean
def dog_check(mystring):
	return 'dog' in mystring.lower()

# if starts with a vowel then add 'ay', otherwise move 1st letter to end and add ay (string concat concatenate)
def pig_latin(word):
	first_letter = word[0]
	# check if vowel
	if first_letter in 'aeiou':
		pig_word = word + 'ay'
	else:
		pig_word = word[1:] + first_letter + 'ay'
	return pig_word

# what if you want to pass parameters? any problems?
def myfunc(a, b, c=0,d=0):
	# a and b are positional arguments, passed in as a tuple with only 2 parameters
	# return 5% of the sum of (a + b)
	return sum(   (a,b,c,d)  ) * 0.05     # entered as a tuple

# arbitrary number of arguments: can now pass in as many arguments as i want
# *args is just a convention, you can do *blahblah
def myfunc(*args):
	return sum(args) * 0.05
	print args # looks just like a tuple, the * term allows you to pass in as many as you want, can loop or aggregate it
	
myfunc(10,20,30,40)

# arbitrary number of key word arguments, returns back a dictionary (of key value pairs)
# can do whatever you want inside your function with a list of dictionary items
# **kwargs is an arbitrary choice, but the two asterix is what's indicated to python
def myfunc(**kwargs):
	if 'fruit' in kwargs:
		print('My fruit of choice is {}'.format(kwargs['fruit']))
	else:
		print('I did not find any fruit here')

myfunc(fruit='apple', veggie = 'lettuce') # output = 'My fruit of choice is apple'

# accepting both arguments, great for outside libraries, careful of ordering
def myfunc(*args,**kwargs):
	print(args)
	print(kwargs)
	print ('I would like {} {}'.format(args[0],kwargs['food']))

myfunc(10,20,30,fruit='orange',food='eggs',animal='dog') # output = 'I would like 10 eggs'


# test to only display even
def myfunc(*args):
    mylist = []
    for x in range(0,len(args)):
        if (args[x] % 2) == 0:
            mylist.append(args[x])
    return mylist

# string modification with *args 
# if even then upper case, if odd then lower case
def myfunc(*args):
    mystring = ""
    tempstring = ""
    for x in range(0,len(args[0])):
        tempstring = args[0][x]
        if x % 2 == 0:
            mystring = mystring + tempstring.upper()
        else:
            mystring = mystring + tempstring.lower()
    return mystring

# how to use sum
def blackjack(a, b, c):
	if sum([a,b,c]) <= 21:
		return sum([a,b,c])
	elif 11 in [a,b,c] and sum([a,b,c])-10 <= 21:
		return sum([a,b,c]) - 10
	else:
		return "Bust"

# how to use while loops and breaks
def summer(array_numbers):
	num = 0
	adder = True
	for num in array_numbers:
		while adder:
			if num != 6:
				total += num
				break
			else:
				adder = False
		while not adder:
			if num != 9:
				break
			else:
				adder = True
				break
	return total
#usage:
summer([1, 3, 5, 7])


#################################################################################################################
# how to enumerate an array with identifiers
###########################
# enumerate the first 10 messages (0 to 9)
for message_no, message in enumerate(messages[:10]):
	print(message_no, message)
	print('\n')
		
		
		
		
		
#################################################################################################################
# how to pop off array items
###########################
def codewords(nums):
	code = [1, 2, 3, 'x']
	# [2, 3, 'x']
	# [3, 'x']
	# ['x']
	for num in nums:
		if num == code[0]:
			code.pop(0)
	return len(code) == 1

# for... else combo
# find primes
def count_primes(num):
	
	# check for 0 and 1
	if num < 2: return 0
	
	# storage
	primes = [2]
	x = 3
	
	while x <= num:
		for y in range(3, x, 2):
			if x%y == 0:
				x += 2				# skip ahead of even numbers
				break
		else:						# this only runs if it doesn't break out of "for"
			primes.append(x)
			x += 2
	print(primes)
	return len(primes)
	

#################################################################################################################
# pandas dataframe basics
		
import pandas as pd
df = pd.read_csv('some_csv.csv')
df.columns # shows columns 
df.index
len(df.index) # number of rows
len(df.columns) # number of columns
df.info() # for all the info

# add a column using apply method (this function converts Yes to 1 and else to 0)
def converter(private_school):
    if private_school == 'Yes':
        return 1
    else:
        return 0
df['Cluster'] = df['Private'].apply(converter) 
		
# look at one entry and edit it
example_entry = personnel['Email'].iloc[0]
df.loc['Cazenovia College', 'Grad.Rate']=100 # sets that value to 100 (do not use the .at function will give slicer dataframe warning)
# use lambda function to split email		
example_entry.split('@')[1] # would give hotmail.com
host = personnel['Email'].apply(lambda email: email.split('@')[1]) # would also give hotmail.com

# filter comparison AND
count_amex_and_above95 = df[(df['CC Provider']=='AMEX') & df['Purchase Price'] > 95)].count()
		
# unique count of classifications (already sorted)
df['Job Title'].value_counts()

# lambda filter on a text string in CC Exp Date for last 2 digits of '10/25' so 3: = 25
sum(df['CC Exp Date'].apply(lambda exp: exp[3:]=='25')) # 1033
df[ df['CC Exp Date'].apply(lambda exp: exp[3:]=='25')].count() # 1033
df[ df['CC Exp Date'].apply(lambda exp: exp[3:]=='25')].value_counts().head(5)

# lambda expression with IF statement to add new column to dataframe 
df['loan_repaid'] = df['loan_status'].apply(lambda status: status=='Fully Paid')  # returns column of True/False
df['loan_repaid'] = df['loan_status'].apply(lambda x : True if (x > 30 and x < 20) else False) # returns column of True/False
df['loan_repaid'] = df['loan_status'].apply(lambda x : 1 if (x == 'Fully Paid') else 0) # returns column of 1/0
df.drop('loan_status', axis=1, inplace=True) # remove text row

# another way to create dummy columns (without lambda)
df['verification_status'].value_counts() # see what the columns will be
statuses = pd.get_dummies(df['verification_status'], drop_first=True) # drop first for machine learning but can remove


#################################################################################################################
# pandas groupby

# group by to see data
# Can also pass column names as group keys
df.groupby('column1_type').mean()

# Make some arrays for use as keys
# Remember np.array(my_list) creates a numpy array
cities = np.array(['NY','LA','LA','NY','NY'])
month = np.array(['JAN','FEB','JAN','FEB','JAN'])
#Now using the data from dataset1, group the means by city and month
df['dataset1'].groupby([cities,month]).mean()		
		
# more group by examples on rows and columns
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
animals = DataFrame(np.arange(16).reshape(4, 4),
                   columns=['W', 'X', 'Y', 'Z'],
                   index=['Dog', 'Cat', 'Bird', 'Mouse'])
# Now modify certain entries with some NAN values
animals.loc[1:2, ['W', 'Y']] = np.nan 
animals
		
# Map Rows Index (Dog, Cat, Bird, Mouse) to Categories
animal_map = {'Dog': 'house', 'Cat': 'house', 'Bird': 'nest','Mouse': 'hole'}
# Now let's try it with a Series
animal_series = Series(animal_map)
# Now let's groupby the Series
animals.groupby(animal_series, axis=0).sum()
		
# Map Columns (W, X, Y, Z) to Categories
behavior_map = {'W': 'good', 'X': 'bad', 'Y': 'good', 'Z': 'bad'}
# Now let's try it with a Series
behav_series = Series(behavior_map)
# Now let's groupby the Series
animals.groupby(behav_series, axis=1).count()

		
#################################################################################################################
# pandas melt
df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
                   'B': {0: 1, 1: 3, 2: 5},
                   'C': {0: 2, 1: 4, 2: 6}})
"""		
df
   A  B  C
0  a  1  2
1  b  3  4
2  c  5  6
"""
pd.melt(df, id_vars=['A'], value_vars=['B'])

"""
   A variable  value
0  a        B      1
1  b        B      3
2  c        B      5
"""

#################################################################################################################		
# Pandas DataFrames - Join/Merge/Pivot Table Recommender System		

# read specific columns
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)
		
# get list of movie titles		
movie_titles = pd.read_csv("Movie_Id_Titles")
movie_titles.head()
		
# merge two dataframes together using Movie ID (to get the text titles and ratings)
df = pd.merge(df,movie_titles,on='item_id')
df.head()		
		
# create a dataframe with average ratings and sort them and take a look at the top of the data
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()		
df.groupby('title')['rating'].count().sort_values(ascending=False).head()
# average ratings dataframe
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()
# number of ratings dataframe
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()

# visualize outlier data in number of ratings - notice a lot of films only had one person rate (distribution skewed left x-axis)
sns.set_style('white')
plt.figure(figsize=(15,8))
ratings['num of ratings'].hist(bins=70)
# visualize outlier data in average ratings - notice a lot of films only had one person rate (concentration on ends, 1 and 5)
plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)
# create a joint plot to see how the ratings and number of ratings are distributed
sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)

# create recommendation matrix using simple correlations by user_id
# Step 1: the matrix will have all the users (each row), all titles (each column), and ratings (in the values) 
moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head() # creates a massive dataset = 944 rows × 1664 columns (excel would explode)
# understand your data - sort the values to see the highest number of ratings (ascending = False)
ratings.sort_values('num of ratings', ascending=False).head(10)

# Step 2: select top two movies and make recommendation - if someone likes this movie what else they could like (by highest corr)
# get all users and Star Wars rating (would have some NaN if no rating)
starwars_user_ratings = moviemat['Star Wars (1977)'] # rows = all users, one column = star wars rating
# get all users and Liar Liar rating (would have some NaN if no rating)
liarliar_user_ratings = moviemat['Liar Liar (1997)']
# so the .corrwith creates a correlation against two dataframes
# Compute pairwise correlation between rows or columns of DataFrame with rows or columns of Series or DataFrame.  
# DataFrames are first aligned along both axes before computing the correlations. 
# 944 rows × 1664 columns vs. 944 rows x 1 columns
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)  # returns 1664 rows movies and 1 correlation number in a Series
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)
similar_to_starwars # title and correlation in a series
# Put series into a dataframe
corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
# Clean up the dataframe by dropping the NaN
corr_starwars.dropna(inplace=True)
corr_starwars.head()

# Step 3: sort data by highest correlation and pull out results that do not make sense
corr_starwars.sort_values('Correlation',ascending=False).head(10)
# look back to chart and see that the cut off is around 100 reviews, so add the number of ratings to the correlation
# the join method requires that the Index is the same (in this case it is, it is the Title)
corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()
# apply the filter (note ascending=false to get highest first)
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()
# do the same for liar liar lovers
corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()



#################################################################################################################
# lambda functions for dummies

# Example 1
x = lambda a : a + 10
print(x(5))   # 15

# Example 2a
# Compare Function vs. Lambda
# put it all in one line, get rid of return, get rid of parenthesis, get rid of name
def cube(y): return y*y*y; 
g =       lambda x: x*x*x 
print(cube(5)) # 125
print(g(7)) # 343

# Example 2b
# Compare if even
even = lambda num: num%2 == 0
even(4) # True
even(3) # False

# Return first letter of string
first = lambda s: s[0]
first('asdf') # a

# Return reverse order of string
reversal = lambda s: s[::-1]
reversal('asdf') # fdsa

# Multiple entries
addermulti = lambda x,y: x+y
addermulti(3,3)
# Multi variables
x = lambda a, b : a * b
print(x(5, 6))  # 30

# Example 3
# Python code to illustrate filter() with lambda() 
li = [5, 7, 22, 97, 54, 62, 77, 23, 73, 61] 
final_list = list(filter(lambda x: (x%2 != 0) , li)) 
print(final_list) # [5, 7, 97, 77, 23, 73, 61]

# Example 4 - removal of illegal character - return position 1 and onwards if first entry has # ex) #EVENTID else line
text_string.map(lambda line: line[1:] if line[0] == '#' else line)
# string tokenization token split 
text_string.map(lambda line: line.split())
      
# Example 5
# map() with lambda()  
# to get double of a list. 
print("map with lambda function: ")
li = [5, 7, 22, 97, 54, 62, 77, 23, 73, 61] 
final_list = list(map(lambda x: x*2 , li)) 
print(final_list)   # [10, 14, 44, 194, 108, 124, 154, 46, 146, 122]

# Example 6
# another map example
x = [2, 3, 4, 5, 6]
y = map(lambda v : v * 5, x)
print(list(y))
		
# Example 7 - to use in Spark
clean = text_string.map(lambda line: line[1:] if line[0] == '#' else line)
# create a TUPLE grab State and Amount
step1 = clean.map(lambda lst: (lst[3], lst[-1])   )
# ReKey = Reduce by Key (like GroupBy) and convert to Float so the + works
step2 = step1.reduceByKey(lambda amt1,amt2 : float(amt1) + float(amt2))
# Get rid of title (first line)
step3 = step2.filter(lambda x: not x[0]=='State')
# Sort Results by Amount
step4 = step3.sortBy(lambda stAmount: stAmount[1],ascending=False)
# Perform the action
step4.collect()
	      
# late binding closures issue in python
# Five functions are created; instead all of them just multiply x by 4
print("with late binding closure: ")
def create_multipliers():
    return [lambda x : i * x for i in range(5)]      # () seems to fix? return (lambda x : i * x for i in range(5))
for multiplier in create_multipliers():
    print(multiplier(2))
    # five 8 are printed: 8, 8, 8, 8, 8

# late binding closures work around 
print("with late binding closure fixed: ")
def create_multipliers_fix():
    return [lambda x, i=i : i * x for i in range(5)]
for multiplier in create_multipliers_fix():
    print(multiplier(2))
    # 0, 2, 4, 6,8

# creates iterable and skip the function
print("another iterable lambda: ")
another = (lambda i: i + x for x in range(5))
for multiplier in another:
    print(multiplier(1))		

"""
with late binding closure: 
8
8
8
8
8
with late binding closure fixed: 
0
2
4
6
8
another iterable lambda: 
1
2
3
4
5
"""		
		
#################################################################################################################
# Load an image
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)		
		
		
		
		
		
		
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################

		
# hi





