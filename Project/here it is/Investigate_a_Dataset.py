#!/usr/bin/env python
# coding: utf-8

# > **Tip**: Welcome to the Investigate a Dataset project! You will find tips in quoted sections like this to help organize your approach to your investigation. Once you complete this project, remove these **Tip** sections from your report before submission. First things first, you might want to double-click this Markdown cell and change the title so that it reflects your dataset and investigation.
# 
# # Project: Investigate a Dataset - [No-show appointments]
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# 
# This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. 
# A number of characteristics about the patient are included in each row.
# 
# ● ‘ScheduledDay’ tells us on what day the patient set up their appointment.
# 
# ● ‘Neighborhood’ indicates the location of the hospital.
# 
# ● ‘Scholarship’ indicates whether or not the patient is enrolled in Brasilian welfare program Bolsa Família.
# 
# ● Be careful about the encoding of the last column: it says ‘No’ if the patient showed up to their appointment, and ‘Yes’ if they did not show up.
# 
# We need to figure out what factors are important for us to predict if the patient will show up for their scheduled appointment?

# In[ ]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > **Tip**: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you **document your data cleaning steps in mark-down cells precisely and justify your cleaning decisions.**
# 
# 
# ### General Properties
# > **Tip**: You should _not_ perform too many operations in each cell. Create cells freely to explore your data. One option that you can take with this project is to do a lot of explorations in an initial notebook. These don't have to be organized, but make sure you use enough comments to understand the purpose of each code cell. Then, after you're done with your analysis, create a duplicate notebook where you will trim the excess and organize your steps so that you have a flowing, cohesive report.

# In[ ]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
df.head()


# In[ ]:


df.shape()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# 
# ### Data Cleaning
# > **Tip**: Make sure that you keep your reader informed on the steps that you are taking in your investigation. Follow every code cell, or every set of related code cells, with a markdown cell to describe to the reader what was found in the preceding cell(s). Try to make it so that the reader can then understand what they will be seeing in the following cell(s).
#  

# In[ ]:


# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.
df.drop(['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay'], axis = 1, inplace = True)
df.head()


# In[ ]:


#Renaming the no-show column
df.rename(columns={'No-show' : 'No_show'}, inplace=True)
df.head()


# In[ ]:


#Correction of "Hipertension" spelling
df.rename(columns={'Hipertension':'Hypertension'},inplace=True)
df.head()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# Let's have a look on the data

# In[ ]:


# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.
df.hist(figsize=(18,10));


# Most of the patient don't have chronic diseases nor beening handicapped.
# 200000 from 110000 have Hypertension.
# The number of patient who received an sms is double of who didn't.
# About 9% are enrolled in the Brasilian welfare program.
# 

# In[ ]:


show = df.No_show =='No'
noshow = df.No_show == 'yes'


# In[ ]:


df[show].count()


# In[ ]:


df[noshow].count()


# The number of those who showed at the clinc was about 4 times whose who didn't.

# ### Research Question 2  (No-show appointments!)

# In[ ]:


plt.figure(figsize=[15.78,7.27])
df.Gender[show].hist(alpha=0.5,label = 'noshow')
plt.legend()
plt.title('comparisonbetween those who showed to those who did not according to Gender')
plt.xlabel('Gender')
plt.ylabel('Patients Number')


# In[ ]:


#compare between who showed to those who didn't in terms of Gender
print(df.Gender[show].value_counts())
print(df.Gender[noshow].value_counts())


# Gender is isigneficant ,Females who showed were more than males who did , and females who did not show were also more than males who did not.

# In[ ]:


#Compare those who showed to those who didn't according to enrollment in the Brasillian welfare program
plt.figure(figsize=[14.78,8.27])
df.Scholarship[show].hist(alpha=0.5,label='show')
df.Scholarship[noshow].hist(alpha=0.5,label='noshow')
plt.legend()
plt.title('Comparsion between those who showed to those who didn;t according to enrollment in the Brasillian welfare program')
plt.xlabel('welfare')
plt.ylabel('Patients Number');


# In[ ]:


#compare between who showed to those who didn't in terms of Gender
print(df.Scholarship[show].value_counts())
print(df.Scholarship[noshow].value_counts())


# Being enrolled in the brasillian welfare is insignificant.

# In[ ]:


#Compare between those who showed to those who didn't according to Hypertension.
plt.figure(figsize=[14.78,8.27])
df.Hipertension[show].hist(alpha=0.5,label='show')
df.Hipertension[noshow].hist(alpha=0.5,label='noshow')
plt.legend()
plt.title('Comparsion between those who showed to those who didnt according to Hypertension')
plt.xlabel('Hypertension')
plt.ylabel('Patients Number');


# Hypertension is insignificant.

# In[ ]:


#Compare between those who showed to those who didn't according to diabetes.
plt.figure(figsize=[14.78,8.27])
df.Diabetes[show].hist(alpha=0.5,label='show')
df.Diabetes[noshow].hist(alpha=0.5,label='noshow')
plt.legend()
plt.title('Comparsion between those who showed to those who didnt according to Diabetes')
plt.xlabel('Diabetes')
plt.ylabel('Patients Number');


# Diabetes is insignificant.

# In[ ]:


#Compare between those who showed to those who didn't according to Alcoholism.
plt.figure(figsize=[14.78,8.27])
df.Alcoholism[show].hist(alpha=0.5,label='show')
df.Alcoholism[noshow].hist(alpha=0.5,label='noshow')
plt.legend()
plt.title('Comparsion between those who showed to those who didnt according to Alcoholism')
plt.xlabel('Alcoholism')
plt.ylabel('Patients Number');


# Alcoholism is insignificant.

# In[ ]:


#Compare between those who showed to those who didn't according to Handicapped.
plt.figure(figsize=[14.78,8.27])
df.Handcap[show].hist(alpha=0.5,label='show')
df.Handcap[noshow].hist(alpha=0.5,label='noshow')
plt.legend()
plt.title('Comparsion between those who showed to those who didnt according to Handcapped')
plt.xlabel('handcapped')
plt.ylabel('Patients Number');


# Handcap is insignificant.

# In[ ]:


#Compare between those who showed to those who didn't according to the SMS.
plt.figure(figsize=[14.78,8.27])
df.SMS_received[show].hist(alpha=0.5,label='show')
df.SMS_received[noshow].hist(alpha=0.5,label='noshow')
plt.legend()
plt.title('Comparsion between those who showed to those who didnt according to SMS Received')
plt.xlabel('Recieved SMS')
plt.ylabel('Patients Number');


# people who have received an SMS message showed less than who did get an SMS.

# In[ ]:


#Compare between those who showed to those who didn't according to their Age.
plt.figure(figsize=[14.78,8.27])
df.Age[show].hist(alpha=0.5,label='show')
df.Age[noshow].hist(alpha=0.5,label='noshow')
plt.legend()
plt.title('Comparsion between those who showed to those who didnt according to Age')
plt.xlabel('Age')
plt.ylabel('Patients Number');


# kids from 0-10 years old showed more than all other ages then from 35-70.

# In[ ]:


#Compare between those who showed to those who didn't according to their neibourhood.
plt.figure(figsize=[14.78,8.27])
df.neighbourhood[show].hist(alpha=0.5,label='show')
df.neighbourhood[noshow].hist(alpha=0.5,label='noshow')
plt.legend()
plt.title('Comparsion between those who showed to their neighbourhood')
plt.xlabel('neighbourhood')
plt.ylabel('Patients Number');


# The Neighbourhood factor is highly effective in showing at the clinics.

# <a id='conclusions'></a>
# ## Conclusions
# At the end , i can say that the neighbourhood has the highest effect on the attendance of the patient to the clinics
# 
# Age is also a good factor as young ages seems to show there more than elders.
# 
# The less people receive SMS messages the more they show up .
# 
# 
# Limitation : Could not detect direct correlation between patients showing and many characteristics such as gender , chronic dieseases and disabilities.
# 
# ## Submitting your Project 
# 
# > **Tip**: Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > **Tip**: Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > **Tip**: Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

