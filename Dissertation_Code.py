#!/usr/bin/env python
# coding: utf-8

# # Importing Data
# 

# In[13]:


import os
import pandas as pd

folder_path = "/Users/pranshu/Desktop/DissData/OneDrive_2_27-06-2024/Engage/Datasets"
dataframes = {}

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_name = os.path.splitext(filename)[0]
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        dataframes[file_name] = df

for key, df in dataframes.items():
    print(f"DataFrame for {key}:")
    print(df.head())


# In[14]:


import os
import pandas as pd

folder_path = "/Users/pranshu/Desktop/DissData/OneDrive_2_27-06-2024/Engage/Datasets"
dataframes = {}

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_name = os.path.splitext(filename)[0]
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        globals()[file_name] = df


# In[15]:


T10_election_voters.head()


# In[17]:


membership_2122=pd.read_csv("membership_2122.csv")


# In[19]:


membership_2223=pd.read_csv("membership_2223.csv")
membership_2324=pd.read_csv("membership2324.csv")


# In[20]:


membership_2122.head()


# In[22]:


eats_reg=pd.read_excel("eats_reg.xlsx")


# In[168]:


eats_txns=pd.read_excel("eats_txns.xlsx")


# In[24]:


eats_reg.head(1)


# In[169]:


eats_txns.head(1)


# In[157]:


fixr_events=pd.read_excel("fixr_events.xlsx")


# In[158]:


fixr_events.columns


# In[161]:


fixr_events['Event Open Time'] = pd.to_datetime(fixr_events['Event Open Time'])

# Group by the specified columns and sum the 'Ticket Quantity' to get 'total_tickets_sold'
event_summary = fixr_events.groupby(['Event Name', 'Event Open Time', 'Venue ID', 'Venue Name'])['Ticket Quantity'].sum().reset_index(name='total_tickets_sold')



# In[162]:


event_summary.head()


# # Membership EDA
# 

# In[26]:


# Number of students who voted
num_students_voted = T10_election_voters['Student ID'].nunique()
print(f"Number of students who voted: {num_students_voted}")

# Number of students who are part of a society
num_students_in_society = T1_clubs_membership['Student ID'].nunique()
print(f"Number of students who are part of a society: {num_students_in_society}")

# Number of students who made an LUU events booking
num_students_luu_booking = T8_luu_events_booking['Student ID'].nunique()
print(f"Number of students who made an LUU events booking: {num_students_luu_booking}")

# Number of students who joined a club
num_students_joined_club = T1_clubs_membership['Student ID'].nunique()
print(f"Number of students who joined a club: {num_students_joined_club}")

# Number of unique events in T4_event_booking
num_unique_events_T4 = T4_event_booking['Event ID'].nunique()
print(f"Number of unique events in T4_event_booking: {num_unique_events_T4}")

# Number of unique events in T7_events_by_luu
num_unique_events_T7 = T7_events_by_luu['Event ID'].nunique()
print(f"Number of unique events in T7_events_by_luu: {num_unique_events_T7}")

# Number of unique events in T5_reslife_events
num_unique_events_T5 = T5_reslife_events['Event ID'].nunique()
print(f"Number of unique events in T5_reslife_events: {num_unique_events_T5}")

# Number of students who made a reslife event booking
num_students_reslife_booking = T6_reslife_event_booking['Student ID'].nunique()
print(f"Number of students who made a reslife event booking: {num_students_reslife_booking}")

# Number of unique societies
num_unique_societies = T1_clubs_membership['Society ID'].nunique()
print(f"Number of unique societies: {num_unique_societies}")

# Number of unique elections
num_unique_elections = T9_luu_elections['Election ID'].nunique()
print(f"Number of unique elections: {num_unique_elections}")

# Number of students who sought advice
num_students_advice = T11_advice_students['Student ID'].nunique()
print(f"Number of students who sought advice: {num_students_advice}")


# In[45]:


# Merging T8_luu_events_booking with T7_events_by_luu to include 'Event Date'
merged_event_data = T8_luu_events_booking.merge(T7_events_by_luu[['Event ID', 'Event Date']], on='Event ID', how='left')

# Check if 'Event Date' is added successfully and view some sample data
print(merged_event_data[['Event ID', 'Event Date']].head())

# Handling the Date for Trend Analysis
merged_event_data['Event Date'] = pd.to_datetime(merged_event_data['Event Date'], errors='coerce')
print("Data types after merging and converting dates:")
print(merged_event_data.dtypes)

# Drop rows where 'Event Date' is NaT (not a time)
merged_event_data = merged_event_data.dropna(subset=['Event Date'])

# Participation Trends Over Time for LUU Events Team
event_trend = merged_event_data.groupby(merged_event_data['Event Date'].dt.to_period('M'))['Student ID'].nunique()

# Plotting the trend
if not event_trend.empty:
    plt.figure(figsize=(12, 8))
    event_trend.plot()
    plt.title('Monthly Participation Trends (LUU Events Team)')
    plt.xlabel('Month')
    plt.ylabel('Number of Participants')
    plt.show()
else:
    print("No data available for trend analysis after handling dates.")


# In[54]:


# Merge the event details from T5 with the booking data from T6
merged_data = T5_reslife_events.merge(T6_reslife_event_booking, on='Event ID')

# Count the number of bookings for each event
event_popularity = merged_data['Event Name_y'].value_counts().head(10)

# Plot the top 10 most popular events
plt.figure(figsize=(12, 8))
sns.barplot(y=event_popularity.index, x=event_popularity.values, palette='viridis')
plt.title('Top 10 Most Popular Events by University Accommodation')
plt.xlabel('Number of Bookings')
plt.ylabel('Event Name')
plt.show()


# In[55]:


# Top N events to display
top_n = 10

# Overview of event participation by event name for LUU Clubs and Societies (Table 4)
event_counts_t4 = T4_event_booking['Event Name'].value_counts().head(top_n)
plt.figure(figsize=(12, 8))
sns.barplot(x=event_counts_t4.values, y=event_counts_t4.index, palette="viridis")
plt.title(f'Top {top_n} Event Participation by Event Name (LUU Clubs and Societies)')
plt.xlabel('Number of Participants')
plt.ylabel('Event Name')
plt.show()


# In[57]:


# Merge T8_luu_events_booking with T7_events_by_luu to get event names
event_booking_with_names = T8_luu_events_booking.merge(T7_events_by_luu[['Event ID', 'Event Name']], on='Event ID', how='left')

# Top N events to display
top_n = 10

# Overview of event participation by event name for LUU Events Team (Table 7 and 8 merged)
event_counts_t8 = event_booking_with_names['Event Name_x'].value_counts().head(top_n)
plt.figure(figsize=(12, 8))
sns.barplot(x=event_counts_t8.values, y=event_counts_t8.index, palette="viridis")
plt.title(f'Top {top_n} Event Participation by Event Name (LUU Events Team)')
plt.xlabel('Number of Participants')
plt.ylabel('Event Name')
plt.show()


# In[59]:


# Top N events to display
top_n = 15

# Combine counts from all event data
all_event_counts = pd.concat([
    T4_event_booking['Event Name'].value_counts(),
    event_booking_with_names['Event Name_x'].value_counts(),  # Assuming this is from T7 & T8 merged
    merged_data['Event Name_y'].value_counts()  # Assuming this is from T5 & T6 merged
])

# Sum the counts for events with the same name
total_event_counts = all_event_counts.groupby(all_event_counts.index).sum()

# Get the top 15 most popular events by number of participants
top_15_events = total_event_counts.nlargest(top_n)

# Plot the top 15 most popular events
plt.figure(figsize=(14, 10))
sns.barplot(y=top_15_events.index, x=top_15_events.values, palette="viridis")
plt.title('Top 15 Most Popular Events Across All Categories')
plt.xlabel('Number of Bookings')
plt.ylabel('Event Name')
plt.show()


# In[61]:


# Merging T8_luu_events_booking with T7_events_by_luu to include 'Event Date'
merged_t7_t8 = T8_luu_events_booking.merge(T7_events_by_luu[['Event ID', 'Event Date']], on='Event ID', how='left')

# Merging T6_reslife_event_booking with T5_reslife_events to include 'Event Date'
merged_t5_t6 = T6_reslife_event_booking.merge(T5_reslife_events[['Event ID', 'Event Date']], on='Event ID', how='left')

# Ensure 'Event Date' is in datetime format
merged_t7_t8['Event Date'] = pd.to_datetime(merged_t7_t8['Event Date'], errors='coerce')
merged_t5_t6['Event Date'] = pd.to_datetime(merged_t5_t6['Event Date'], errors='coerce')
T4_event_booking['Event Date'] = pd.to_datetime(T4_event_booking['Event Date'], errors='coerce')

# Combine all event data with 'Event Date' and 'Student ID'
all_events_data = pd.concat([
    T4_event_booking[['Event Date', 'Student ID']],
    merged_t7_t8[['Event Date', 'Student ID']],
    merged_t5_t6[['Event Date', 'Student ID']]
])

# Drop any rows where 'Event Date' might be NaT due to conversion errors
all_events_data = all_events_data.dropna(subset=['Event Date'])



# In[62]:


# Group by month and count unique student IDs for each month
monthly_participation = all_events_data.groupby(all_events_data['Event Date'].dt.to_period('M'))['Student ID'].nunique()

# Plot the monthly participation trends
plt.figure(figsize=(14, 7))
monthly_participation.plot(kind='bar')
plt.title('Monthly Participation Trends Across All Event Types')
plt.xlabel('Month')
plt.ylabel('Number of Unique Participants')
plt.xticks(rotation=90)  # Ensuring month labels are readable
plt.show()


# In[73]:


# Group by month and count unique student IDs for each month
weekly_participation = all_events_data.groupby(all_events_data['Event Date'].dt.to_period('W'))['Student ID'].nunique()

# Plot the monthly participation trends
plt.figure(figsize=(14, 7))
monthly_participation.plot(kind='line')
plt.title('Weekly Participation Trends Across All Event Types')
plt.xlabel('Week')
plt.ylabel('Number of Unique Participants')
plt.xticks(rotation=90)  # Ensuring month labels are readable
plt.show()


# # Eats EDA

# In[63]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Basic statistics for registrations
print("Basic Statistics for Registrations:")
print(eats_reg.describe(include='all'))

# Basic statistics for transactions
print("Basic Statistics for Transactions:")
print(eats_txns.describe(include='all'))

# Number of unique users
unique_users = eats_reg['STUDENT_ID_NUMBER'].nunique()
print(f"Number of unique users: {unique_users}")

# Histogram of registration dates
eats_reg['DataRegistered'] = pd.to_datetime(eats_reg['DataRegistered'])
eats_reg['DataRegistered'].hist(bins=50)
plt.title('Histogram of Registration Dates')
plt.xlabel('Date')
plt.ylabel('Number of Registrations')
plt.show()


# In[65]:


# Convert the 'DateRegistered' column to datetime for accurate plotting
eats_reg['DateRegistered'] = pd.to_datetime(eats_reg['DateRegistered'])

# Plotting the histogram of registration dates
plt.figure(figsize=(10, 6))
eats_reg['DateRegistered'].hist(bins=50)
plt.title('Histogram of Registration Dates')
plt.xlabel('Date')
plt.ylabel('Number of Registrations')
plt.show()


# In[66]:


# Plotting the frequency of first use sites
plt.figure(figsize=(10, 6))
sns.countplot(y='FirstUseSite', data=eats_reg, order=eats_reg['FirstUseSite'].value_counts().index)
plt.title('Popularity of First Use Sites')
plt.xlabel('Count')
plt.ylabel('First Use Site')
plt.show()


# In[68]:


# Assuming 'TransactionDate' is a column in 'eats_txns' and has been converted to datetime format
eats_txns['TransactionDate'] = pd.to_datetime(eats_txns['TransactionDate'])
monthly_transactions = eats_txns.set_index('TransactionDate').resample('M').size()

# Plotting monthly transaction trends
plt.figure(figsize=(14, 7))
monthly_transactions.plot(kind='bar')
plt.title('Monthly Transaction Trends')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=90)  # Rotate labels for better readability
plt.show()


# In[145]:


# Basic summary statistics for numerical data
print("Summary statistics for 'Amount':")
print(eats_txns['Amount'].describe())

# Frequency count for categorical data like Transaction Type
print("Transaction Type Counts:")
print(eats_txns['TransactionType'].value_counts())


# In[146]:


# Plotting the distribution of transaction types
sns.countplot(x='TransactionType', data=eats_txns)
plt.title('Distribution of Transaction Types')
plt.xlabel('Transaction Type')
plt.ylabel('Frequency')
plt.show()


# In[147]:


# Convert TransactionDate to datetime
eats_txns['TransactionDate'] = pd.to_datetime(eats_txns['TransactionDate'])

# Plotting transaction volume over time
eats_txns.set_index('TransactionDate')['Amount'].resample('M').sum().plot(kind='line')
plt.title('Monthly Transaction Volume')
plt.xlabel('Month')
plt.ylabel('Total Amount Spent')
plt.show()


# In[72]:


# Assuming 'TransactionDate' is a column in 'eats_txns' and has been converted to datetime format
eats_txns['TransactionDate'] = pd.to_datetime(eats_txns['TransactionDate'])
monthly_transactions = eats_txns.set_index('TransactionDate').resample('W').size()

# Plotting monthly transaction trends
plt.figure(figsize=(14, 7))
monthly_transactions.plot(kind='line')
plt.title('Weekly Transaction Trends')
plt.xlabel('Week')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=90)  # Rotate labels for better readability
plt.show()


# Avg txns/user havent increased as much as the txns, this shows that there are new users coming to the app, but the old users are not sticking around, which ideally should be the aim for a loyalty porgram. 

# In[151]:


# Convert TransactionDate to datetime
eats_txns['TransactionDate'] = pd.to_datetime(eats_txns['TransactionDate'])

# Set TransactionDate as the index
eats_txns.set_index('TransactionDate', inplace=True)

# Group by CardID and resample to weekly, counting the number of transactions per week
weekly_txns_per_user = eats_txns.groupby('CardID').resample('W').size()

# Reset index to plot data
weekly_txns_per_user = weekly_txns_per_user.reset_index(name='Transactions')

# Plot the average number of transactions per user per week
weekly_txns_per_user_average = weekly_txns_per_user.groupby('TransactionDate')['Transactions'].mean()

plt.figure(figsize=(10, 6))
weekly_txns_per_user_average.plot()
plt.title('Average Weekly Transactions Per User')
plt.xlabel('Week')
plt.ylabel('Average Number of Transactions')
plt.grid(True)
plt.show()



# In[148]:


# Group by 'Site' and sum 'Amount'
site_transaction_totals = eats_txns.groupby('Site')['Amount'].sum().sort_values(ascending=False)

# Plotting
site_transaction_totals.plot(kind='bar')
plt.title('Total Spend by Outlet')
plt.xlabel('Outlet')
plt.ylabel('Total Spend')
plt.show()


# #### Rewards are mostly collected by staff members and this is a clear indication to show that students are not aware of the app and/or are not using it at all. 

# In[149]:


# Count of each type of reward claimed
reward_counts = eats_txns['Reward'].value_counts()

# Plotting reward redemption frequency
reward_counts.plot(kind='bar')
plt.title('Frequency of Reward Redemptions')
plt.xlabel('Reward')
plt.ylabel('Count')
plt.show()


# In[150]:


# Boxplot to show distribution of amounts by transaction type
sns.boxplot(x='TransactionType', y='Amount', data=eats_txns)
plt.title('Transaction Amounts by Transaction Type')
plt.xlabel('Transaction Type')
plt.ylabel('Amount')
plt.show()


# ## Event Participation and Txns correlation: LUU Events followed by Fixr Events

# In[69]:


import pandas as pd
import matplotlib.pyplot as plt

# Convert transaction and event dates to datetime if not already
eats_txns['TransactionDate'] = pd.to_datetime(eats_txns['TransactionDate'])
all_events_data['Event Date'] = pd.to_datetime(all_events_data['Event Date'])

# Resample transactions to weekly and count the number of transactions
weekly_transactions = eats_txns.set_index('TransactionDate').resample('W').size()

# Resample event participation to weekly and count unique student IDs
weekly_participants = all_events_data.drop_duplicates(subset=['Student ID', 'Event Date'])\
                                     .set_index('Event Date')\
                                     .resample('W')['Student ID'].nunique()

# Ensure both series cover the same period for comparison
start_date = max(weekly_transactions.index.min(), weekly_participants.index.min())
end_date = min(weekly_transactions.index.max(), weekly_participants.index.max())

weekly_transactions = weekly_transactions[start_date:end_date]
weekly_participants = weekly_participants[start_date:end_date]


# In[70]:


fig, ax1 = plt.subplots(figsize=(14, 7))

color = 'tab:red'
ax1.set_xlabel('Week')
ax1.set_ylabel('Transactions', color=color)
ax1.plot(weekly_transactions.index, weekly_transactions.values, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Unique Participants', color=color)  # we already handled the x-label with ax1
ax2.plot(weekly_participants.index, weekly_participants.values, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # to ensure the layout isn't too cramped
plt.title('Weekly Transactions and Event Participation')
plt.show()


# In[197]:


import matplotlib.pyplot as plt
import pandas as pd

# Apply a style for nicer grid and background color
plt.style.use('seaborn-darkgrid')

# Convert transaction and event dates to datetime if not already
eats_txns['TransactionDate'] = pd.to_datetime(eats_txns['TransactionDate'])
all_events_data['Event Date'] = pd.to_datetime(all_events_data['Event Date'])

# Resample transactions to weekly and count the number of transactions
weekly_transactions = eats_txns.set_index('TransactionDate').resample('W').size()

# Apply rolling mean to smooth transaction counts
smoothed_weekly_transactions = weekly_transactions.rolling(window=3, center=True).mean()

# Resample event participation to weekly and count unique student IDs
weekly_participants = all_events_data.drop_duplicates(subset=['Student ID', 'Event Date'])\
                                     .set_index('Event Date')\
                                     .resample('W')['Student ID'].nunique()

# Apply rolling mean to smooth participant counts
smoothed_weekly_participants = weekly_participants.rolling(window=3, center=True).mean()

# Ensure both series cover the same period for comparison
start_date = max(smoothed_weekly_transactions.index.min(), smoothed_weekly_participants.index.min())
end_date = min(smoothed_weekly_transactions.index.max(), smoothed_weekly_participants.index.max())

smoothed_weekly_transactions = smoothed_weekly_transactions[start_date:end_date]
smoothed_weekly_participants = smoothed_weekly_participants[start_date:end_date]

fig, ax1 = plt.subplots(figsize=(14, 7))

# Plotting transactions
color = 'tab:red'
ax1.set_xlabel('Week',fontsize=20)
ax1.set_ylabel('Transactions', color=color,fontsize=20)
ax1.plot(smoothed_weekly_transactions.index, smoothed_weekly_transactions.values, color=color, marker='o', linestyle='-', linewidth=2, markersize=5)
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelcolor=color,labelsize=18)

# Plotting unique participants
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Registrations', color=color,fontsize=20)
ax2.plot(smoothed_weekly_participants.index, smoothed_weekly_participants.values, color=color, marker='o', linestyle='-', linewidth=2, markersize=5)
ax2.tick_params(axis='y', labelcolor=color,labelsize=18)

fig.tight_layout()
plt.title('Weekly Transactions and LUU Event Registration',fontsize=20)
plt.show()


# In[172]:


# If 'Event Open Time' is tz-aware and you want to remove the timezone
event_summary['Event Open Time'] = event_summary['Event Open Time'].dt.tz_localize(None)


# In[173]:


# Convert 'Event Open Time' to datetime format
event_summary['Event Open Time'] = pd.to_datetime(event_summary['Event Open Time'])

# Resample to get the total tickets sold per week
weekly_event_tickets = event_summary.set_index('Event Open Time').resample('W')['total_tickets_sold'].sum()

# Convert 'TransactionDate' to datetime and resample to weekly
eats_txns['TransactionDate'] = pd.to_datetime(eats_txns['TransactionDate'])
weekly_transactions = eats_txns.set_index('TransactionDate').resample('W').size()

# Define start and end dates based on both datasets
start_date = max(weekly_transactions.index.min(), weekly_event_tickets.index.min())
end_date = min(weekly_transactions.index.max(), weekly_event_tickets.index.max())

# Restrict data to the common date range
weekly_transactions = weekly_transactions[start_date:end_date]
weekly_event_tickets = weekly_event_tickets[start_date:end_date]

fig, ax1 = plt.subplots(figsize=(14, 7))

color = 'tab:red'
ax1.set_xlabel('Week')
ax1.set_ylabel('Transactions', color=color)
ax1.plot(weekly_transactions.index, weekly_transactions.values, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Tickets Sold', color=color)  # we already handled the x-label with ax1
ax2.plot(weekly_event_tickets.index, weekly_event_tickets.values, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # to ensure the layout isn't too cramped
plt.title('Weekly Transactions and Ticket Sales from Fixr Events')
plt.show()



# In[192]:


import matplotlib.pyplot as plt
import pandas as pd

# Apply a style
plt.style.use('seaborn-darkgrid')  # This applies a nice grid and background color

# Smoothing the lines by applying a rolling mean
rolling_window_size = 3  # You can adjust this window size
smoothed_weekly_transactions = weekly_transactions.rolling(window=rolling_window_size, center=True).mean()
smoothed_weekly_event_tickets = weekly_event_tickets.rolling(window=rolling_window_size, center=True).mean()

fig, ax1 = plt.subplots(figsize=(14, 7))

# Plotting transactions
color = 'tab:red'
ax1.set_xlabel('Week', fontsize=20)
ax1.set_ylabel('Transactions', color=color, fontsize=20)
ax1.plot(smoothed_weekly_transactions.index, smoothed_weekly_transactions.values, color=color, marker='o', linestyle='-', linewidth=2, markersize=5)
ax1.tick_params(axis='x', labelsize=16)  # Adjust font size for x-axis labels
ax1.tick_params(axis='y', labelcolor=color, labelsize=20)

# Plotting tickets sold
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Tickets Sold', color=color, fontsize=20)
ax2.plot(smoothed_weekly_event_tickets.index, smoothed_weekly_event_tickets.values, color=color, marker='o', linestyle='-', linewidth=2, markersize=5)
ax2.tick_params(axis='y', labelcolor=color, labelsize=18)

fig.tight_layout()
plt.title('Weekly Transactions and Ticket Sales from Fixr Events', fontsize=20)
plt.show()


# In[179]:


import matplotlib.pyplot as plt
import pandas as pd

# Apply a style
plt.style.use('seaborn-darkgrid')  # This applies a nice grid and background color

# Smoothing the lines by applying a rolling mean
rolling_window_size = 3  # You can adjust this window size
smoothed_monthly_transactions = monthly_transactions.rolling(window=rolling_window_size, center=True).mean()
smoothed_monthly_event_tickets = monthly_event_tickets.rolling(window=rolling_window_size, center=True).mean()

fig, ax1 = plt.subplots(figsize=(14, 7))

# Plotting transactions
color = 'tab:red'
ax1.set_xlabel('Month')  # Changed from 'Week' to 'Month' for accuracy
ax1.set_ylabel('Transactions', color=color)
ax1.plot(smoothed_monthly_transactions.index, smoothed_monthly_transactions.values, color=color, marker='o', linestyle='-', linewidth=2, markersize=5)
ax1.tick_params(axis='y', labelcolor=color)

# Plotting tickets sold
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Tickets Sold', color=color)
ax2.plot(smoothed_monthly_event_tickets.index, smoothed_monthly_event_tickets.values, color=color, marker='o', linestyle='-', linewidth=2, markersize=5)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Monthly Transactions and Ticket Sales from Fixr Events (Smoothed)')
plt.show()


# ## Creating a combined DF with event participation and Transactions

# In[82]:


# Correcting the column name in the eats_reg DataFrame to match eats_txns
eats_reg.rename(columns={'CardId': 'CardID'}, inplace=True)

# Merge the dataframes on CardID now that the names are aligned
combined_txns = pd.merge(eats_txns, eats_reg[['CardID', 'STUDENT_ID_NUMBER']], on='CardID', how='left')


print(combined_txns.head())


# In[95]:


len(combined_txns.STUDENT_ID_NUMBER.dropna())


# In[96]:


len(combined_txns.STUDENT_ID_NUMBER)


# In[97]:


len(combined_txns.STUDENT_ID_NUMBER.unique())


# In[83]:


# Group transaction data by Student ID to get total amount spent and count of transactions
transaction_summary = combined_txns.groupby('STUDENT_ID_NUMBER').agg({
    'Amount': 'sum',
    'TransactionId': 'count'
}).rename(columns={'TransactionId': 'Transaction Count', 'Amount': 'Total Spent'})


print(transaction_summary.head())


# In[84]:


# Aggregate event data by Student ID
event_participation_summary = all_events_data.groupby('Student ID').size().reset_index(name='Event Count')


print(event_participation_summary.head())


# In[102]:


event_participation_summary[event_participation_summary['Student ID']=='01094507']


# In[111]:


event_participation_summary.dtypes


# In[114]:


transaction_summary.reset_index()['STUDENT_ID_NUMBER'].unique()


# In[117]:


transaction_summary.reset_index()['STUDENT_ID_NUMBER'].astype(str)


# In[134]:


event_participation_summary=event_participation_summary.drop(columns='STUDENT_ID_NUMBER')


# In[135]:


# Merge event participation data with transaction summary
transaction_summary=transaction_summary.reset_index()
event_participation_summary['Student ID']=event_participation_summary['Student ID'].astype(str)
transaction_summary['STUDENT_ID_NUMBER']=transaction_summary['STUDENT_ID_NUMBER'].astype(str)

combined_data = pd.merge(event_participation_summary, transaction_summary, left_on='Student ID', right_on='STUDENT_ID_NUMBER', how='left')

# Fill missing transaction data with 0 (assuming no transactions means 0 spent and 0 transactions)
combined_data.fillna({'Total Spent': 0, 'Transaction Count': 0}, inplace=True)

# Display the combined data
print(combined_data.head())


# In[130]:


import statsmodels.api as sm

# Prepare the independent variables (X) and the dependent variable (y)
X = combined_data['Event Count']  # Independent variable: the number of events participated in
y = combined_data['Total Spent']  # Dependent variable: total amount spent

# Adding a constant to the model (intercept)
X = sm.add_constant(X)

# Fit an OLS regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression
print(model.summary())


# In[136]:


combined_data.head(100)


# In[141]:


combined_data[combined_data['Total Spent']>0]['Transaction Count'].sum()


# In[137]:


# Check if all 'Total Spent' and 'Transaction Count' are zero
all_zeros_spent = combined_data['Total Spent'].eq(0).all()
all_zeros_count = combined_data['Transaction Count'].eq(0).all()

print(f"All Total Spent are zeros: {all_zeros_spent}")
print(f"All Transaction Count are zeros: {all_zeros_count}")


# In[90]:


# Check the distribution of 'Event Count'
event_distribution = event_participation_summary['Event Count'].value_counts()

# Number of students attending more than one event
more_than_one_event = event_participation_summary[event_participation_summary['Event Count'] > 1].shape[0]

# Number of students attending only one event
only_one_event = event_participation_summary[event_participation_summary['Event Count'] == 1].shape[0]

print(f"Number of students attending more than one event: {more_than_one_event}")
print(f"Number of students attending only one event: {only_one_event}")
print(event_distribution)


# In[91]:


# Count unique STUDENT_ID_NUMBER in eats_reg
unique_ids_eats_reg = eats_reg['STUDENT_ID_NUMBER'].nunique()
print(f"Unique STUDENT_ID_NUMBER in eats_reg: {unique_ids_eats_reg}")

# Count unique Student ID in all_events_data
unique_ids_all_events = all_events_data['Student ID'].nunique()
print(f"Unique Student ID in all_events_data: {unique_ids_all_events}")


# In[92]:


# Convert ID columns to same type if necessary
eats_reg['STUDENT_ID_NUMBER'] = eats_reg['STUDENT_ID_NUMBER'].astype(str)
all_events_data['Student ID'] = all_events_data['Student ID'].astype(str)

# Merge to find matching Student IDs
matched_ids = pd.merge(eats_reg[['STUDENT_ID_NUMBER']], all_events_data[['Student ID']], left_on='STUDENT_ID_NUMBER', right_on='Student ID', how='inner')

# Count of unique matched Student IDs
unique_matched_ids = matched_ids['STUDENT_ID_NUMBER'].nunique()
print(f"Number of matching Student IDs: {unique_matched_ids}")


# In[106]:


matched_ids.head()


# In[93]:


all_events_data.head()


# In[94]:


eats_reg.head()


# In[107]:


eats_reg['CardID'].unique()


# In[108]:


eats_txns.dtypes


# In[109]:


eats_reg.dtypes


# In[110]:


eats_txns['CardID'].unique()


# In[142]:


filtered_ids = eats_reg[eats_reg['STUDENT_ID_NUMBER'].astype(str).str.match(r'^2\d{8}$')]

print(filtered_ids['STUDENT_ID_NUMBER'])


# In[143]:


count_ids = filtered_ids['STUDENT_ID_NUMBER'].count()


# In[144]:


print(count_ids)


# ### App Download and Usage Funnel

# Total Number of App Downloads:
# 
# iOS: 8000
# Android: 829+789+46+4+2 = 1670 
# 
# Total: 9670 (100% after accounting for total number of people the app is exposed to)
# 
# Total Registatrions on the Eats App: 3953 (40% conversion from downloads)
# 
# Total Number of People Transacting After Registering: 2664 (27% conversion from downloads, 67% conversion from regs)

# In[153]:


eats_txns['CardID'].nunique()


# In[ ]:




