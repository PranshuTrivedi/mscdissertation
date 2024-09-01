#!/usr/bin/env python
# coding: utf-8

# # Importing Data
# 

# In[1]:


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


# In[2]:


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


# In[3]:


T10_election_voters.head()


# In[4]:


membership_2122=pd.read_csv("membership_2122.csv")


# In[5]:


membership_2223=pd.read_csv("membership_2223.csv")
membership_2324=pd.read_csv("membership2324.csv")


# In[6]:


membership_2122.head()


# In[7]:


eats_reg=pd.read_excel("eats_reg.xlsx")


# In[8]:


eats_txns=pd.read_excel("eats_txns.xlsx")


# In[9]:


eats_reg.head(1)


# In[10]:


eats_reg.rename(columns={'CardId': 'CardID'}, inplace=True)


# In[11]:


eats_txns.head(1)


# In[12]:


eats_txns['TransactionDate'].min()


# In[13]:


eats_txns['TransactionDate'].max()


# In[14]:


fixr_events=pd.read_excel("fixr_events.xlsx")


# In[15]:


fixr_events.columns


# In[16]:


fixr_events['Event Open Time'] = pd.to_datetime(fixr_events['Event Open Time'])

# Group by the specified columns and sum the 'Ticket Quantity' to get 'total_tickets_sold'
event_summary = fixr_events.groupby(['Event Name', 'Event Open Time', 'Venue ID', 'Venue Name'])['Ticket Quantity'].sum().reset_index(name='total_tickets_sold')



# In[17]:


event_summary.head()


# # Membership EDA
# 

# In[18]:


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


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


# In[20]:


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


# In[21]:


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


# In[22]:


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


# In[23]:


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


# In[24]:


T4_event_booking.columns


# In[25]:


T4_event_booking=pd.read_csv("T4_with_dates.csv")


# In[26]:


T4_event_booking.columns


# In[27]:


T4_event_booking['Event Date'] = pd.to_datetime(T4_event_booking['Event Date'], format='%d/%m/%y',dayfirst=True)


# In[28]:


T4_event_booking.head()


# In[29]:


# Merging T8_luu_events_booking with T7_events_by_luu to include 'Event Date'
merged_t7_t8 = T8_luu_events_booking.merge(T7_events_by_luu[['Event ID', 'Event Date']], on='Event ID', how='left')

# Merging T6_reslife_event_booking with T5_reslife_events to include 'Event Date'
merged_t5_t6 = T6_reslife_event_booking.merge(T5_reslife_events[['Event ID', 'Event Date']], on='Event ID', how='left')

# Ensure 'Event Date' is in datetime format
merged_t7_t8['Event Date'] = pd.to_datetime(merged_t7_t8['Event Date'], errors='coerce')
merged_t5_t6['Event Date'] = pd.to_datetime(merged_t5_t6['Event Date'], errors='coerce')
#T4_event_booking['Event Date'] = pd.to_datetime(T4_event_booking['Event Date'], errors='coerce')

# Combine all event data with 'Event Date' and 'Student ID'
all_events_data = pd.concat([
    T4_event_booking[['Event Date', 'Student ID']],
    merged_t7_t8[['Event Date', 'Student ID']],
    merged_t5_t6[['Event Date', 'Student ID']]
])

# Drop any rows where 'Event Date' might be NaT due to conversion errors
all_events_data = all_events_data.dropna(subset=['Event Date'])



# In[30]:


merged_t7_t8.head(10)


# In[31]:


T4_event_booking["Event Date"].isna().sum()


# In[32]:


all_events_data.head()


# In[33]:


all_events_data["Event Date"].max()


# In[34]:


T4_event_booking["Event Date"].min()


# In[35]:


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


# In[36]:


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

# In[37]:


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


# In[39]:


# Convert the 'DateRegistered' column to datetime for accurate plotting
eats_reg['DateRegistered'] = pd.to_datetime(eats_reg['DateRegistered'])

# Plotting the histogram of registration dates
plt.figure(figsize=(10, 6))
eats_reg['DateRegistered'].hist(bins=50)
plt.title('Histogram of Registration Dates')
plt.xlabel('Date')
plt.ylabel('Number of Registrations')
plt.show()


# In[40]:


# Plotting the frequency of first use sites
plt.figure(figsize=(10, 6))
sns.countplot(y='FirstUseSite', data=eats_reg, order=eats_reg['FirstUseSite'].value_counts().index)
plt.title('Popularity of First Use Sites')
plt.xlabel('Count')
plt.ylabel('First Use Site')
plt.show()


# In[41]:


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


# In[42]:


# Basic summary statistics for numerical data
print("Summary statistics for 'Amount':")
print(eats_txns['Amount'].describe())

# Frequency count for categorical data like Transaction Type
print("Transaction Type Counts:")
print(eats_txns['TransactionType'].value_counts())


# In[43]:


# Plotting the distribution of transaction types
sns.countplot(x='TransactionType', data=eats_txns)
plt.title('Distribution of Transaction Types')
plt.xlabel('Transaction Type')
plt.ylabel('Frequency')
plt.show()


# In[44]:


# Convert TransactionDate to datetime
eats_txns['TransactionDate'] = pd.to_datetime(eats_txns['TransactionDate'])

# Plotting transaction volume over time
eats_txns.set_index('TransactionDate')['Amount'].resample('M').sum().plot(kind='line')
plt.title('Monthly Transaction Volume')
plt.xlabel('Month')
plt.ylabel('Total Amount Spent')
plt.show()


# In[45]:


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

# In[46]:


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



# In[47]:


# Group by 'Site' and sum 'Amount'
site_transaction_totals = eats_txns.groupby('Site')['Amount'].sum().sort_values(ascending=False)

# Plotting
site_transaction_totals.plot(kind='bar')
plt.title('Total Spend by Outlet')
plt.xlabel('Outlet')
plt.ylabel('Total Spend')
plt.show()


# #### Rewards are mostly collected by staff members and this is a clear indication to show that students are not aware of the app and/or are not using it at all. 

# In[48]:


# Count of each type of reward claimed
reward_counts = eats_txns['Reward'].value_counts()

# Plotting reward redemption frequency
reward_counts.plot(kind='bar')
plt.title('Frequency of Reward Redemptions')
plt.xlabel('Reward')
plt.ylabel('Count')
plt.show()


# In[49]:


# Boxplot to show distribution of amounts by transaction type
sns.boxplot(x='TransactionType', y='Amount', data=eats_txns)
plt.title('Transaction Amounts by Transaction Type')
plt.xlabel('Transaction Type')
plt.ylabel('Amount')
plt.show()


# ## Event Participation and Txns correlation: LUU Events followed by Fixr Events

# In[50]:


eats_txns.columns


# In[51]:


eats_txns=pd.read_excel("eats_txns.xlsx")


# In[52]:


print(eats_reg.columns)
print(eats_txns.columns)


# In[53]:


import pandas as pd
import matplotlib.pyplot as plt

# Ensure the columns exist and are of the correct type
eats_txns['CardID'] = eats_txns['CardID'].astype(str)
eats_reg['CardID'] = eats_reg['CardID'].astype(str)

# Merge eats_txns with eats_reg to get 'STUDENT_ID_NUMBER' in eats_txns_merged
eats_txns_merged = eats_txns.merge(eats_reg[['CardID', 'STUDENT_ID_NUMBER']], on='CardID', how='left')

# Identify students who participated in at least one event
event_participants = set(all_events_data['Student ID'].unique())

# Classify transactions as made by participants or non-participants
eats_txns_merged['ParticipantStatus'] = eats_txns_merged['STUDENT_ID_NUMBER'].apply(
    lambda x: 'Participant' if x in event_participants else 'Non-Participant'
)

# Filter data from April 2023 onwards
eats_txns_merged = eats_txns_merged[eats_txns_merged['TransactionDate'] >= '2023-04-01']

# Resample transactions to weekly for both groups
weekly_transactions_participants = eats_txns_merged[eats_txns_merged['ParticipantStatus'] == 'Participant']\
    .set_index('TransactionDate').resample('W').size()

weekly_transactions_non_participants = eats_txns_merged[eats_txns_merged['ParticipantStatus'] == 'Non-Participant']\
    .set_index('TransactionDate').resample('W').size()

# Ensure both series cover the same period for comparison
start_date = max(weekly_transactions_participants.index.min(), weekly_transactions_non_participants.index.min())
end_date = min(weekly_transactions_participants.index.max(), weekly_transactions_non_participants.index.max())

weekly_transactions_participants = weekly_transactions_participants[start_date:end_date]
weekly_transactions_non_participants = weekly_transactions_non_participants[start_date:end_date]

# Smoothing the lines by applying a rolling mean
rolling_window_size = 3
smoothed_weekly_transactions_participants = weekly_transactions_participants.rolling(window=rolling_window_size, center=True).mean()
smoothed_weekly_transactions_non_participants = weekly_transactions_non_participants.rolling(window=rolling_window_size, center=True).mean()

# Apply a style
plt.style.use('seaborn-darkgrid')

fig, ax1 = plt.subplots(figsize=(14, 7))

# Plotting participant transactions
ax1.set_xlabel('Week', fontsize=20)
ax1.set_ylabel('Transactions', fontsize=20)
ax1.plot(smoothed_weekly_transactions_participants.index, smoothed_weekly_transactions_participants.values, color='tab:blue', marker='o', linestyle='-', linewidth=2, markersize=5, label='Participant Transactions')

# Plotting non-participant transactions
ax1.plot(smoothed_weekly_transactions_non_participants.index, smoothed_weekly_transactions_non_participants.values, color='tab:red', marker='o', linestyle='-', linewidth=2, markersize=5, label='Non-Participant Transactions')

ax1.tick_params(axis='x', labelsize=16)  # Adjust font size for x-axis labels
ax1.tick_params(axis='y', labelsize=20)

# Adding a legend
ax1.legend(fontsize=16)

fig.tight_layout()
plt.title('Weekly Transactions: Participants vs Non-Participants', fontsize=20)
plt.show()


# In[54]:


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
ax2.set_ylabel('Participants', color=color,fontsize=20)
ax2.plot(smoothed_weekly_participants.index, smoothed_weekly_participants.values, color=color, marker='o', linestyle='-', linewidth=2, markersize=5)
ax2.tick_params(axis='y', labelcolor=color,labelsize=18)

fig.tight_layout()
plt.title('Weekly Transactions and LUU Event Participants',fontsize=20)
plt.show()


# In[55]:


# If 'Event Open Time' is tz-aware and you want to remove the timezone
event_summary['Event Open Time'] = event_summary['Event Open Time'].dt.tz_localize(None)


# In[56]:


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



# In[57]:


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


# ## Creating a combined DF with event participation and Transactions

# In[58]:


# Correcting the column name in the eats_reg DataFrame to match eats_txns
eats_reg.rename(columns={'CardId': 'CardID'}, inplace=True)

# Merge the dataframes on CardID now that the names are aligned
combined_txns = pd.merge(eats_txns, eats_reg[['CardID', 'STUDENT_ID_NUMBER']], on='CardID', how='left')


print(combined_txns.head())


# In[59]:


len(combined_txns.STUDENT_ID_NUMBER.dropna())


# In[60]:


len(combined_txns.STUDENT_ID_NUMBER)


# In[61]:


len(combined_txns.STUDENT_ID_NUMBER.unique())


# In[62]:


# Group transaction data by Student ID to get total amount spent and count of transactions
transaction_summary = combined_txns.groupby('STUDENT_ID_NUMBER').agg({
    'Amount': 'sum',
    'TransactionId': 'count'
}).rename(columns={'TransactionId': 'Transaction Count', 'Amount': 'Total Spent'})


print(transaction_summary.head())


# In[63]:


# Aggregate event data by Student ID
event_participation_summary = all_events_data.groupby('Student ID').size().reset_index(name='Event Count')


print(event_participation_summary.head())


# In[64]:


event_participation_summary[event_participation_summary['Student ID']=='01094507']


# In[65]:


event_participation_summary.dtypes


# In[66]:


transaction_summary.reset_index()['STUDENT_ID_NUMBER'].unique()


# In[67]:


transaction_summary.reset_index()['STUDENT_ID_NUMBER'].astype(str)


# In[68]:


event_participation_summary=event_participation_summary.drop(columns='STUDENT_ID_NUMBER')


# In[69]:


# Merge event participation data with transaction summary
transaction_summary=transaction_summary.reset_index()
event_participation_summary['Student ID']=event_participation_summary['Student ID'].astype(str)
transaction_summary['STUDENT_ID_NUMBER']=transaction_summary['STUDENT_ID_NUMBER'].astype(str)

combined_data = pd.merge(event_participation_summary, transaction_summary, left_on='Student ID', right_on='STUDENT_ID_NUMBER', how='left')

# Fill missing transaction data with 0 (assuming no transactions means 0 spent and 0 transactions)
combined_data.fillna({'Total Spent': 0, 'Transaction Count': 0}, inplace=True)

# Display the combined data
print(combined_data.head())


# In[70]:


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


# In[71]:


combined_data.head(100)


# In[72]:


combined_data[combined_data['Total Spent']>0]['Transaction Count'].sum()


# In[73]:


# Check if all 'Total Spent' and 'Transaction Count' are zero
all_zeros_spent = combined_data['Total Spent'].eq(0).all()
all_zeros_count = combined_data['Transaction Count'].eq(0).all()

print(f"All Total Spent are zeros: {all_zeros_spent}")
print(f"All Transaction Count are zeros: {all_zeros_count}")


# In[74]:


# Check the distribution of 'Event Count'
event_distribution = event_participation_summary['Event Count'].value_counts()

# Number of students attending more than one event
more_than_one_event = event_participation_summary[event_participation_summary['Event Count'] > 1].shape[0]

# Number of students attending only one event
only_one_event = event_participation_summary[event_participation_summary['Event Count'] == 1].shape[0]

print(f"Number of students attending more than one event: {more_than_one_event}")
print(f"Number of students attending only one event: {only_one_event}")
print(event_distribution)


# In[75]:


# Count unique STUDENT_ID_NUMBER in eats_reg
unique_ids_eats_reg = eats_reg['STUDENT_ID_NUMBER'].nunique()
print(f"Unique STUDENT_ID_NUMBER in eats_reg: {unique_ids_eats_reg}")

# Count unique Student ID in all_events_data
unique_ids_all_events = all_events_data['Student ID'].nunique()
print(f"Unique Student ID in all_events_data: {unique_ids_all_events}")


# In[76]:


# Convert ID columns to same type if necessary
eats_reg['STUDENT_ID_NUMBER'] = eats_reg['STUDENT_ID_NUMBER'].astype(str)
all_events_data['Student ID'] = all_events_data['Student ID'].astype(str)

# Merge to find matching Student IDs
matched_ids = pd.merge(eats_reg[['STUDENT_ID_NUMBER']], all_events_data[['Student ID']], left_on='STUDENT_ID_NUMBER', right_on='Student ID', how='inner')

# Count of unique matched Student IDs
unique_matched_ids = matched_ids['STUDENT_ID_NUMBER'].nunique()
print(f"Number of matching Student IDs: {unique_matched_ids}")


# In[77]:


# Convert ID columns to same type if necessary
eats_reg['STUDENT_ID_NUMBER'] = eats_reg['STUDENT_ID_NUMBER'].astype(str)
all_events_data['Student ID'] = all_events_data['Student ID'].astype(str)

# Merge to find matching Student IDs for registration
matched_reg_ids = pd.merge(eats_reg[['STUDENT_ID_NUMBER']], all_events_data[['Student ID']], left_on='STUDENT_ID_NUMBER', right_on='Student ID', how='inner')

# Count of unique matched Student IDs for registration
unique_matched_reg_ids = matched_reg_ids['STUDENT_ID_NUMBER'].nunique()

# Total unique Student IDs in all_events_data
total_event_participants = all_events_data['Student ID'].nunique()

# Percentage of students who registered for LUU Eats from those who attended an event
percentage_registered = (unique_matched_reg_ids / total_event_participants) * 100
print(f"Percentage of students who registered for LUU Eats from those who attended an event: {percentage_registered:.2f}%")


# In[78]:


# Convert 'CardID' column in eats_txns to same type as 'CardID' in eats_reg
eats_txns['CardID'] = eats_txns['CardID'].astype(str)
eats_reg['CardID'] = eats_reg['CardID'].astype(str)

# Merge eats_txns with eats_reg to get the STUDENT_ID_NUMBER for transactions
transactions_with_ids = pd.merge(eats_txns[['CardID']], eats_reg[['CardID', 'STUDENT_ID_NUMBER']], on='CardID', how='left')

# Filter out transactions without a matching STUDENT_ID_NUMBER
transactions_with_ids = transactions_with_ids.dropna(subset=['STUDENT_ID_NUMBER'])

# Ensure STUDENT_ID_NUMBER is of type string
transactions_with_ids['STUDENT_ID_NUMBER'] = transactions_with_ids['STUDENT_ID_NUMBER'].astype(str)

# Merge to find matching Student IDs for transactions
matched_txn_ids = pd.merge(transactions_with_ids[['STUDENT_ID_NUMBER']], all_events_data[['Student ID']], left_on='STUDENT_ID_NUMBER', right_on='Student ID', how='inner')

# Count of unique matched Student IDs for transactions
unique_matched_txn_ids = matched_txn_ids['STUDENT_ID_NUMBER'].nunique()

# Percentage of students who made a transaction on the Eats app from those who attended an event
percentage_transacted = (unique_matched_txn_ids / total_event_participants) * 100
print(f"Percentage of students who made a transaction on the Eats app from those who attended an event: {percentage_transacted:.2f}%")


# In[79]:


unique_matched_txn_ids


# In[80]:


import matplotlib.pyplot as plt

# Data to plot
registered_percentage = 11.18
transacted_percentage = 4.74
not_registered_or_transacted_percentage = 100 - registered_percentage - transacted_percentage

labels = 'Registered for LUU Eats', 'Made a Transaction', 'Did not register or make a transaction'
sizes = [registered_percentage, transacted_percentage, not_registered_or_transacted_percentage]
colors = ['#ff9999','#66b3ff','#99ff99']
explode = (0.1, 0.1, 0)  # explode 1st and 2nd slice

# Plotting the pie chart
plt.figure(figsize=(10, 7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140)
plt.title('Engagement of Students Who Attended an Event with LUU Eats App')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show the pie chart
plt.show()


# In[81]:


matched_ids.head()


# In[82]:


all_events_data.head()


# In[83]:


eats_reg.head()


# In[84]:


eats_reg['CardID'].unique()


# In[85]:


eats_txns.dtypes


# In[86]:


eats_reg.dtypes


# In[87]:


eats_txns['CardID'].unique()


# In[88]:


filtered_ids = eats_reg[eats_reg['STUDENT_ID_NUMBER'].astype(str).str.match(r'^2\d{8}$')]

print(filtered_ids['STUDENT_ID_NUMBER'])


# In[89]:


count_ids = filtered_ids['STUDENT_ID_NUMBER'].count()


# In[90]:


print(count_ids)


# In[91]:


# Assuming the relevant column is named 'STUDENT_ID' in both DataFrames

# Combine the 'STUDENT_ID' columns from both DataFrames
combined_student_ids = pd.concat([membership_2223['STUDENT_ID_NUMBER'], membership_2324['STUDENT_ID_NUMBER']])

# Get the unique student IDs from the combined data
unique_students_combined = combined_student_ids.unique()

# Print the number of unique students in the combined data
print(f"Total unique students from both years combined: {len(unique_students_combined)}")


# In[92]:


import pandas as pd

# Ensure 'CardID' is a string
eats_txns['CardID'] = eats_txns['CardID'].astype(str)
eats_reg['CardID'] = eats_reg['CardID'].astype(str)

# Merge eats_txns with eats_reg to get 'STUDENT_ID_NUMBER' in eats_txns_merged
eats_txns_merged = eats_txns.merge(eats_reg[['CardID', 'STUDENT_ID_NUMBER']], on='CardID', how='left')

# Group by 'STUDENT_ID_NUMBER' and count the number of transactions per student
student_txn_counts = eats_txns_merged.groupby('STUDENT_ID_NUMBER').size()

# Filter students with at least 2 transactions
students_with_2_or_more_txns = student_txn_counts[student_txn_counts >= 2]

# Get the number of unique students with at least 2 transactions
num_students_with_2_or_more_txns = len(students_with_2_or_more_txns)

# Print the result
print(f"Number of students with at least 2 transactions: {num_students_with_2_or_more_txns}")


# ### App Download and Usage Funnel

# Total Number of App Downloads:
# 
# iOS: 8000
# 
# Android: 829+789+46+4+2 = 1670 
# 
# Total Members from Oct 22: 51707
# 
# Total Downloads: 9670 
# 
# Total Registatrions on the Eats App: 3953 
# 
# Total Number of People Transacting After Registering: 2664 
# 
# Total Number of People with at least 2 transactions (Retention): 1773

# In[93]:


eats_txns['CardID'].nunique()


# ## Aim-2: Correlations Across Different Types of Participation

# In[94]:


# Ensure relevant columns are strings
eats_reg['STUDENT_ID_NUMBER'] = eats_reg['STUDENT_ID_NUMBER'].astype(str)
eats_txns['CardID'] = eats_txns['CardID'].astype(str)

# Total Unique Voters
total_voters = T10_election_voters['Student ID'].nunique()
print(f"Total unique voters: {total_voters}")

# Total Unique Event Participants
total_event_participants = all_events_data['Student ID'].nunique()
print(f"Total unique event participants: {total_event_participants}")

# Total Unique Students who registered on the eats app
total_eats_registered = eats_reg['STUDENT_ID_NUMBER'].nunique()
print(f"Total unique students who registered on the eats app: {total_eats_registered}")

# Total Unique Students who transacted on the eats app
total_eats_transacted = eats_txns['CardID'].nunique()
print(f"Total unique students who transacted on the eats app: {total_eats_transacted}")

# Merge election voters with eats registration
voters_with_eats = pd.merge(T10_election_voters[['Student ID']], eats_reg[['CardID', 'STUDENT_ID_NUMBER']], left_on='Student ID', right_on='STUDENT_ID_NUMBER', how='inner')
unique_voters_registered = voters_with_eats['Student ID'].nunique()

# Merge voters with transaction data
voters_with_txns = pd.merge(voters_with_eats[['Student ID', 'CardID']], eats_txns[['CardID']], on='CardID', how='inner')
unique_voters_transacted = voters_with_txns['Student ID'].nunique()

# Merge event participants with eats registration
events_with_eats = pd.merge(all_events_data, eats_reg[['CardID', 'STUDENT_ID_NUMBER']], left_on='Student ID', right_on='STUDENT_ID_NUMBER', how='inner')
unique_events_registered = events_with_eats['Student ID'].nunique()

# Merge event participants with transaction data
events_with_txns = pd.merge(events_with_eats[['Student ID', 'CardID']], eats_txns[['CardID']], on='CardID', how='inner')
unique_events_transacted = events_with_txns['Student ID'].nunique()

# Calculate percentages
percent_voters_registered = (unique_voters_registered / total_voters) * 100
percent_voters_transacted = (unique_voters_transacted / total_voters) * 100
percent_events_registered = (unique_events_registered / total_event_participants) * 100
percent_events_transacted = (unique_events_transacted / total_event_participants) * 100

print(f"Percentage of voters who registered for eats: {percent_voters_registered:.2f}%")
print(f"Percentage of voters who transacted on eats: {percent_voters_transacted:.2f}%")
print(f"Percentage of event participants who registered for eats: {percent_events_registered:.2f}%")
print(f"Percentage of event participants who transacted on eats: {percent_events_transacted:.2f}%")


# In[95]:


# Ensure relevant columns are strings
eats_reg['STUDENT_ID_NUMBER'] = eats_reg['STUDENT_ID_NUMBER'].astype(str)
eats_txns['CardID'] = eats_txns['CardID'].astype(str)

# Total Unique Voters
total_voters = T10_election_voters['Student ID'].nunique()
print(f"Total unique voters: {total_voters}")

# Total Unique Event Participants
total_event_participants = all_events_data['Student ID'].nunique()
print(f"Total unique event participants: {total_event_participants}")

# Total Unique Students who registered on the eats app
total_eats_registered = eats_reg['STUDENT_ID_NUMBER'].nunique()
print(f"Total unique students who registered on the eats app: {total_eats_registered}")

# Total Unique Students who transacted on the eats app
total_eats_transacted = eats_txns['CardID'].nunique()
print(f"Total unique students who transacted on the eats app: {total_eats_transacted}")

# Merge election voters with eats registration
voters_with_eats = pd.merge(T10_election_voters[['Student ID']], eats_reg[['CardID', 'STUDENT_ID_NUMBER']], left_on='Student ID', right_on='STUDENT_ID_NUMBER', how='inner')
unique_voters_registered = voters_with_eats['Student ID'].nunique()

# Merge voters with transaction data
voters_with_txns = pd.merge(voters_with_eats[['Student ID', 'CardID']], eats_txns[['CardID', 'TransactionId']], on='CardID', how='inner')
unique_voters_transacted = voters_with_txns['Student ID'].nunique()

# Calculate average number of transactions per voter
transactions_per_voter = voters_with_txns.groupby('Student ID')['TransactionId'].count()
average_transactions_per_voter = transactions_per_voter.mean()
print(f"Average number of transactions per voter: {average_transactions_per_voter:.2f}")

# Identify non-voters
non_voters = eats_reg[~eats_reg['STUDENT_ID_NUMBER'].isin(T10_election_voters['Student ID'])]

# Merge non-voters with transaction data
non_voters_with_txns = pd.merge(non_voters[['STUDENT_ID_NUMBER', 'CardID']], eats_txns[['CardID', 'TransactionId']], on='CardID', how='inner')

# Calculate average number of transactions per non-voter
transactions_per_non_voter = non_voters_with_txns.groupby('STUDENT_ID_NUMBER')['TransactionId'].count()
average_transactions_per_non_voter = transactions_per_non_voter.mean()
print(f"Average number of transactions per non-voter: {average_transactions_per_non_voter:.2f}")

# Merge event participants with eats registration
events_with_eats = pd.merge(all_events_data, eats_reg[['CardID', 'STUDENT_ID_NUMBER']], left_on='Student ID', right_on='STUDENT_ID_NUMBER', how='inner')
unique_events_registered = events_with_eats['Student ID'].nunique()

# Merge event participants with transaction data
events_with_txns = pd.merge(events_with_eats[['Student ID', 'CardID']], eats_txns[['CardID', 'TransactionId']], on='CardID', how='inner')
unique_events_transacted = events_with_txns['Student ID'].nunique()

# Calculate average number of transactions per event participant
transactions_per_event_participant = events_with_txns.groupby('Student ID')['TransactionId'].count()
average_transactions_per_event_participant = transactions_per_event_participant.mean()
print(f"Average number of transactions per event participant: {average_transactions_per_event_participant:.2f}")

# Identify non-participants
non_participants = eats_reg[~eats_reg['STUDENT_ID_NUMBER'].isin(all_events_data['Student ID'])]

# Merge non-participants with transaction data
non_participants_with_txns = pd.merge(non_participants[['STUDENT_ID_NUMBER', 'CardID']], eats_txns[['CardID', 'TransactionId']], on='CardID', how='inner')

# Calculate average number of transactions per non-participant
transactions_per_non_participant = non_participants_with_txns.groupby('STUDENT_ID_NUMBER')['TransactionId'].count()
average_transactions_per_non_participant = transactions_per_non_participant.mean()
print(f"Average number of transactions per non-participant: {average_transactions_per_non_participant:.2f}")

# Calculate percentages
percent_voters_registered = (unique_voters_registered / total_voters) * 100
percent_voters_transacted = (unique_voters_transacted / total_voters) * 100
percent_events_registered = (unique_events_registered / total_event_participants) * 100
percent_events_transacted = (unique_events_transacted / total_event_participants) * 100

print(f"Percentage of voters who registered for eats: {percent_voters_registered:.2f}%")
print(f"Percentage of voters who transacted on eats: {percent_voters_transacted:.2f}%")
print(f"Percentage of event participants who registered for eats: {percent_events_registered:.2f}%")
print(f"Percentage of event participants who transacted on eats: {percent_events_transacted:.2f}%")


# ## Aim-3: Patterns in Participation

# In[96]:


# Ensure ID columns are strings
T10_election_voters['Student ID'] = T10_election_voters['Student ID'].astype(str)
T9_luu_elections['Election ID'] = T9_luu_elections['Election ID'].astype(str)
T10_election_voters['Election ID'] = T10_election_voters['Election ID'].astype(str)

# Merge T10 with T9 to get election dates
election_voters_with_dates = pd.merge(T10_election_voters, T9_luu_elections[['Election ID', 'Election Start Date']], on='Election ID', how='left')

# Convert 'Election Start Date' to datetime
election_voters_with_dates['Election Start Date'] = pd.to_datetime(election_voters_with_dates['Election Start Date'])

# Ensure 'Event Date' is in datetime format
all_events_data['Event Date'] = pd.to_datetime(all_events_data['Event Date'])

# Resample participation data to monthly and count unique participants
monthly_participation_voters = election_voters_with_dates.set_index('Election Start Date').resample('M')['Student ID'].nunique()
monthly_participation_events = all_events_data.set_index('Event Date').resample('M')['Student ID'].nunique()

# Plot the trends
plt.figure(figsize=(14, 7))
plt.plot(monthly_participation_voters.index, monthly_participation_voters.values, label='Voters', marker='o', linestyle='-')
plt.plot(monthly_participation_events.index, monthly_participation_events.values, label='Event Participants', marker='o', linestyle='-')
plt.title('Monthly Participation Trends')
plt.xlabel('Month')
plt.ylabel('Number of Participants')
plt.legend()
plt.show()


# In[97]:


T9_luu_elections.columns


# In[98]:


T10_election_voters.columns


# In[99]:


# Ensure ID columns are strings
T10_election_voters['Student ID'] = T10_election_voters['Student ID'].astype(str)
T1_clubs_membership['Student ID'] = T1_clubs_membership['Student ID'].astype(str)

# Merge election voters with club memberships
voters_with_memberships = pd.merge(T10_election_voters, T1_clubs_membership, left_on='Student ID', right_on='Student ID', how='inner')

# Count unique students who are both voters and members of academic societies
unique_voters_members = voters_with_memberships['Student ID'].nunique()
print(f"Number of unique students who are both voters and members of academic societies: {unique_voters_members}")


# In[100]:


# Assume we already have voters_with_memberships dataframe from previous analyses
# Ensure 'TransactionDate' is in datetime format
eats_txns['TransactionDate'] = pd.to_datetime(eats_txns['TransactionDate'])

# Filter transactions for academic society members who vote
voters_members_txns = pd.merge(voters_with_memberships, eats_reg[['CardID', 'STUDENT_ID_NUMBER']], left_on='Student ID', right_on='STUDENT_ID_NUMBER', how='inner')
voters_members_txns = pd.merge(voters_members_txns, eats_txns, left_on='CardID', right_on='CardID', how='inner')

# Resample transactions to monthly and count unique transactions
monthly_voters_members_txns = voters_members_txns.set_index('TransactionDate').resample('M')['TransactionId'].nunique()

# Plot the trend
plt.figure(figsize=(14, 7))
plt.plot(monthly_voters_members_txns.index, monthly_voters_members_txns.values, label='Transactions by Voters & Members', marker='o', linestyle='-')
plt.title('Monthly Transactions for Academic Society Members Who Vote')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.legend()
plt.show()


# In[101]:


# Ensure 'STUDENT_ID_NUMBER' is a string
eats_reg['STUDENT_ID_NUMBER'] = eats_reg['STUDENT_ID_NUMBER'].astype(str)
T10_election_voters['Student ID'] = T10_election_voters['Student ID'].astype(str)

# Identify voters
voters = T10_election_voters['Student ID'].unique()

# Create voter group transactions
voter_txns = eats_txns[eats_txns['CardID'].isin(eats_reg[eats_reg['STUDENT_ID_NUMBER'].isin(voters)]['CardID'])]

# Create non-voter group transactions
non_voter_txns = eats_txns[~eats_txns['CardID'].isin(eats_reg[eats_reg['STUDENT_ID_NUMBER'].isin(voters)]['CardID'])]

# Resample transactions to monthly for both groups
monthly_voter_txns = voter_txns.set_index('TransactionDate').resample('M')['TransactionId'].nunique()
monthly_non_voter_txns = non_voter_txns.set_index('TransactionDate').resample('M')['TransactionId'].nunique()


# In[102]:


# Get top election dates with the most voters
top_election_dates = election_voters_with_dates['Election Start Date'].value_counts().nlargest(3).index

# Plot the transaction comparison
plt.figure(figsize=(14, 7))

# Plot voter transactions
plt.plot(monthly_voter_txns.index, monthly_voter_txns.values, label='Voter Transactions', marker='o', linestyle='-', color='blue')

# Plot non-voter transactions
plt.plot(monthly_non_voter_txns.index, monthly_non_voter_txns.values, label='Non-Voter Transactions', marker='o', linestyle='-', color='orange')

# Highlight top election dates
for date in top_election_dates:
    plt.axvline(x=date, color='red', linestyle='--', label=f'Election Date: {date.date()}')

plt.title('Monthly Transactions: Voters vs. Non-Voters')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.legend()
plt.show()


# ### Same graph but filtered 

# In[103]:


# Ensure 'STUDENT_ID_NUMBER' is a string
eats_reg['STUDENT_ID_NUMBER'] = eats_reg['STUDENT_ID_NUMBER'].astype(str)
T10_election_voters['Student ID'] = T10_election_voters['Student ID'].astype(str)

# Identify voters
voters = T10_election_voters['Student ID'].unique()

# Create voter group transactions
voter_txns = eats_txns[eats_txns['CardID'].isin(eats_reg[eats_reg['STUDENT_ID_NUMBER'].isin(voters)]['CardID'])]

# Create non-voter group transactions
non_voter_txns = eats_txns[~eats_txns['CardID'].isin(eats_reg[eats_reg['STUDENT_ID_NUMBER'].isin(voters)]['CardID'])]

# Filter transactions to post October 2022
voter_txns = voter_txns[voter_txns['TransactionDate'] >= '2022-10-01']
non_voter_txns = non_voter_txns[non_voter_txns['TransactionDate'] >= '2022-10-01']

# Resample transactions to monthly for both groups
monthly_voter_txns = voter_txns.set_index('TransactionDate').resample('M')['TransactionId'].nunique()
monthly_non_voter_txns = non_voter_txns.set_index('TransactionDate').resample('M')['TransactionId'].nunique()


# In[104]:


# Filter election voters with dates to post October 2022
election_voters_with_dates_filtered = election_voters_with_dates[election_voters_with_dates['Election Start Date'] >= '2022-10-01']

# Get top election dates with the most voters
top_election_dates = election_voters_with_dates_filtered['Election Start Date'].value_counts().nlargest(4).index

# Plot the transaction comparison
plt.figure(figsize=(14, 7))

# Plot voter transactions
plt.plot(monthly_voter_txns.index, monthly_voter_txns.values, label='Voter Transactions', marker='o', linestyle='-', color='blue')

# Plot non-voter transactions
plt.plot(monthly_non_voter_txns.index, monthly_non_voter_txns.values, label='Non-Voter Transactions', marker='o', linestyle='-', color='orange')

# Highlight top election dates
for date in top_election_dates:
    plt.axvline(x=date, color='red', linestyle='--', label=f'Election Date: {date.date()}')

plt.title('Monthly Transactions: Voters vs. Non-Voters')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.legend()
plt.show()


# In[105]:


eats_txns.columns


# In[106]:


eats_reg.columns


# In[107]:


# Merging T8_luu_events_booking with T7_events_by_luu to include 'Event Date' and 'Event Name'
merged_t7_t8 = T8_luu_events_booking.merge(
    T7_events_by_luu[['Event ID', 'Event Date', 'Event Name']], 
    on='Event ID', 
    how='left',
    suffixes=('_booking', '_luu')
)

# Merging T6_reslife_event_booking with T5_reslife_events to include 'Event Date' and 'Event Name'
merged_t5_t6 = T6_reslife_event_booking.merge(
    T5_reslife_events[['Event ID', 'Event Date', 'Event Name']], 
    on='Event ID', 
    how='left',
    suffixes=('_booking', '_reslife')
)

# Ensure 'Event Date' is in datetime format
merged_t7_t8['Event Date'] = pd.to_datetime(merged_t7_t8['Event Date'], errors='coerce')
merged_t5_t6['Event Date'] = pd.to_datetime(merged_t5_t6['Event Date'], errors='coerce')
T4_event_booking['Event Date'] = pd.to_datetime(T4_event_booking['Event Date'], errors='coerce')

# Combine all event data with 'Event Date', 'Student ID', and 'Event Name'
all_events_data_new = pd.concat([
    T4_event_booking[['Event Date', 'Student ID', 'Event Name']],
    merged_t7_t8[['Event Date', 'Student ID', 'Event Name_luu']].rename(columns={'Event Name_luu': 'Event Name'}),
    merged_t5_t6[['Event Date', 'Student ID', 'Event Name_reslife']].rename(columns={'Event Name_reslife': 'Event Name'})
])

# Drop any rows where 'Event Date' might be NaT due to conversion errors
all_events_data_new = all_events_data_new.dropna(subset=['Event Date'])

# Verify the new dataframe
print("all_events_data_new columns:", all_events_data_new.columns)
print(all_events_data_new.head())


# In[108]:


# Ensure 'STUDENT_ID_NUMBER' is a string
eats_reg['STUDENT_ID_NUMBER'] = eats_reg['STUDENT_ID_NUMBER'].astype(str)
T10_election_voters['Student ID'] = T10_election_voters['Student ID'].astype(str)

# Identify voters
voters = T10_election_voters['Student ID'].unique()

# Create voter group transactions
voter_txns = eats_txns[eats_txns['CardID'].isin(eats_reg[eats_reg['STUDENT_ID_NUMBER'].isin(voters)]['CardID'])]

# Create non-voter group transactions
non_voter_txns = eats_txns[~eats_txns['CardID'].isin(eats_reg[eats_reg['STUDENT_ID_NUMBER'].isin(voters)]['CardID'])]

# Filter transactions to post October 2022
voter_txns = voter_txns[voter_txns['TransactionDate'] >= '2022-10-01']
non_voter_txns = non_voter_txns[non_voter_txns['TransactionDate'] >= '2022-10-01']

# Resample transactions to monthly for both groups
monthly_voter_txns = voter_txns.set_index('TransactionDate').resample('M')['TransactionId'].nunique()
monthly_non_voter_txns = non_voter_txns.set_index('TransactionDate').resample('M')['TransactionId'].nunique()


# In[110]:


# Filter all events data to post October 2022
all_events_data_filtered = all_events_data_new[all_events_data_new['Event Date'] >= '2022-10-01']

# Get top event dates with the most participants
top_event_dates = all_events_data_filtered['Event Date'].value_counts().nlargest(5).index
top_event_names = all_events_data_filtered[all_events_data_filtered['Event Date'].isin(top_event_dates)][['Event Date', 'Event Name']].drop_duplicates()

# Create a dictionary for top event dates and their names
top_events_dict = {row['Event Date']: row['Event Name'] for _, row in top_event_names.iterrows()}

# Plot the transaction comparison
plt.figure(figsize=(14, 7))

# Plot voter transactions
plt.plot(monthly_voter_txns.index, monthly_voter_txns.values, label='Voter Transactions', marker='o', linestyle='-', color='blue')

# Plot non-voter transactions
plt.plot(monthly_non_voter_txns.index, monthly_non_voter_txns.values, label='Non-Voter Transactions', marker='o', linestyle='-', color='orange')

# Highlight top event dates with names
for date, name in top_events_dict.items():
    plt.axvline(x=date, color='green', linestyle='--')
    plt.text(date, plt.ylim()[1], '', rotation=90, verticalalignment='top', fontsize=10, color='green')

# Adding event names to legend
for date, name in top_events_dict.items():
    plt.axvline(x=date, color='green', linestyle='--', label=f'Event Date: {date.date()} ({name})')

plt.title('Monthly Transactions: Voters vs. Non-Voters')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.legend()
plt.show()



# ## Voter Event Participation Trend

# In[139]:


# Ensure 'Student ID' is a string in all_events_data_new
all_events_data_new['Student ID'] = all_events_data_new['Student ID'].astype(str)

# Identify event participants who are voters
voter_participants = all_events_data_new[all_events_data_new['Student ID'].isin(voters)]

# Identify event participants who are non-voters
non_voter_participants = all_events_data_new[~all_events_data_new['Student ID'].isin(voters)]


# In[140]:


# Resample event participation to monthly for both groups
monthly_voter_participants = voter_participants.set_index('Event Date').resample('M')['Student ID'].nunique()
monthly_non_voter_participants = non_voter_participants.set_index('Event Date').resample('M')['Student ID'].nunique()

# Plot the event participation comparison
plt.figure(figsize=(14, 7))

# Plot voter event participation
plt.plot(monthly_voter_participants.index, monthly_voter_participants.values, label='Voter Event Participants', marker='o', linestyle='-', color='blue')

# Plot non-voter event participation
plt.plot(monthly_non_voter_participants.index, monthly_non_voter_participants.values, label='Non-Voter Event Participants', marker='o', linestyle='-', color='orange')

plt.title('Monthly Event Participation: Voters vs. Non-Voters')
plt.xlabel('Month')
plt.ylabel('Number of Event Participants')
plt.legend()
plt.show()


# ## Classification Model based on spending trends

# In[141]:


# Ensure CardID is string for consistency
eats_txns['CardID'] = eats_txns['CardID'].astype(str)

# Aggregate the number of transactions per student
transactions_per_student = eats_txns.groupby('CardID').size().reset_index(name='TransactionCount')

# Merge with student demographics from eats_reg to get student IDs
transactions_per_student = transactions_per_student.merge(eats_reg[['CardID', 'STUDENT_ID_NUMBER']], on='CardID', how='left')

# Calculate the quartiles
quartiles = transactions_per_student['TransactionCount'].quantile([0.25, 0.5, 0.75])
print("Quartiles for the number of transactions per student:")
print(quartiles)


# In[142]:


membership_2223.columns


# In[143]:


membership_2223['RESIDENCY_DESC'].head()


# In[144]:


membership_2223['RESIDENCY_DESC'].unique()


# In[145]:


membership_2223['COLL_DESC'].unique()


# In[146]:


membership_2223['YEAR_CODE'].unique()


# In[304]:


import pandas as pd

# Start with membership_2324
combined_membership = membership_2324.copy()

# Identify unique student IDs in membership_2324
student_ids_2324 = set(combined_membership['STUDENT_ID_NUMBER'])

# Exclude student IDs from membership_2223 that are already in membership_2324
membership_2223_filtered = membership_2223[~membership_2223['STUDENT_ID_NUMBER'].isin(student_ids_2324)]

# Combine membership_2324 with the filtered membership_2223
combined_membership = pd.concat([combined_membership, membership_2223_filtered])

# Identify unique student IDs in the combined_membership so far
student_ids_combined = set(combined_membership['STUDENT_ID_NUMBER'])

# Exclude student IDs from membership_2122 that are already in the combined_membership
membership_2122_filtered = membership_2122[~membership_2122['STUDENT_ID_NUMBER'].isin(student_ids_combined)]

# Combine the current combined_membership with the filtered membership_2122
combined_membership = pd.concat([combined_membership, membership_2122_filtered])

# Sort the combined membership by STUDENT_ID_NUMBER and START_DATE
combined_membership_sorted = combined_membership.sort_values(by=['STUDENT_ID_NUMBER', 'START_DATE'], ascending=[True, False])

# Drop duplicates, keeping only the most recent start date for each student
combined_membership = combined_membership_sorted.drop_duplicates(subset=['STUDENT_ID_NUMBER'], keep='first')

# Verify the resulting combined_membership
print("Combined Membership DataFrame after removing duplicates:")
print(combined_membership.head())
print("Number of unique students in combined_membership:", combined_membership['STUDENT_ID_NUMBER'].nunique())
print("Total number of rows in combined_membership:", combined_membership.shape[0])


# ## Note: Mention how combined_membership was created and highlight the challenge of duplicate student_ids

# In[305]:


import pandas as pd


# Extract the required demographic details
demographic_details = combined_membership[['STUDENT_ID_NUMBER', 'GENDER_CODE', 'DATE_OF_BIRTH', 'RESIDENCY_DESC', 'YEAR_CODE', 'FT_PT_IND', 'COLL_DESC']]

# Calculate Age from DATE_OF_BIRTH
current_year = pd.to_datetime('today').year
demographic_details['Age'] = current_year - pd.to_datetime(demographic_details['DATE_OF_BIRTH'], errors='coerce').dt.year

# Classify RESIDENCY_DESC as Home or International
def classify_residency(residency):
    if 'Home' in residency:
        return 'Home'
    else:
        return 'International'

demographic_details['Residency'] = demographic_details['RESIDENCY_DESC'].apply(classify_residency)

# Ensure CardID and STUDENT_ID_NUMBER are strings for consistency
eats_reg['CardID'] = eats_reg['CardID'].astype(str)
eats_reg['STUDENT_ID_NUMBER'] = eats_reg['STUDENT_ID_NUMBER'].astype(str)
demographic_details['STUDENT_ID_NUMBER'] = demographic_details['STUDENT_ID_NUMBER'].astype(str)

# Merge demographic details with eats registration data
students_with_eats = demographic_details.merge(eats_reg[['CardID', 'STUDENT_ID_NUMBER']], on='STUDENT_ID_NUMBER', how='inner')

# Aggregate the number of transactions per student
transactions_per_student = eats_txns.groupby('CardID').size().reset_index(name='TransactionCount')

# Merge with transactions data to get total transactions per student
students_with_eats = students_with_eats.merge(transactions_per_student[['CardID', 'TransactionCount']], on='CardID', how='left')

# Fill NaN values in TransactionCount with 0 (students who registered but made no transactions)
students_with_eats['TransactionCount'] = students_with_eats['TransactionCount'].fillna(0)

# Calculate the quartiles
quartiles = students_with_eats['TransactionCount'].quantile([0.25, 0.5, 0.75])

# Define the classification function based on quartiles
def classify_transaction_count(transaction_count):
    if transaction_count <= quartiles[0.25]:
        return 'Low'
    elif transaction_count <= quartiles[0.75]:
        return 'Medium'
    else:
        return 'High'

# Apply the classification
students_with_eats['TransactionClass'] = students_with_eats['TransactionCount'].apply(classify_transaction_count)

# Final DataFrame with required columns
final_df = students_with_eats[['STUDENT_ID_NUMBER', 'GENDER_CODE', 'Age', 'Residency', 'YEAR_CODE', 'FT_PT_IND', 'COLL_DESC', 'TransactionCount', 'TransactionClass']]

# Define the mapping function for YEAR_CODE
def map_year_code(year_code):
    if 'A' in year_code:
        return 0
    if year_code == '01':
        return 1
    elif year_code == '02':
        return 2
    elif year_code == '03':
        return 3
    elif year_code == '04':
        return 4
    elif year_code == '05':
        return 5
    elif year_code == '06':
        return 6
    elif year_code == '07':
        return 7
    else:
        return int(year_code)  # Convert numeric strings directly to integers

# Apply the mapping function to the YEAR_CODE column
final_df['YEAR_CODE'] = final_df['YEAR_CODE'].apply(map_year_code)

# Verify the changes
print(final_df['YEAR_CODE'].unique())
print(final_df.head())


# In[306]:


combined_membership.columns


# In[307]:


final_df.head()


# In[308]:


# Get the number of unique values in each column
unique_values = final_df.nunique()

# Print the number of unique values for each column
print(unique_values)


# In[309]:


final_df['YEAR_CODE'].unique()


# In[310]:


# Print unique values and their data types for relevant columns
columns_to_check = ['GENDER_CODE', 'Age', 'Residency', 'YEAR_CODE', 'FT_PT_IND', 'COLL_DESC']

for column in columns_to_check:
    unique_values = final_df[column].unique()
    dtype = final_df[column].dtype
    print(f"Column: {column}")
    print(f"Data Type: {dtype}")
    print(f"Unique Values: {unique_values}")
    print("\n")


# In[311]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Fill NaN values in FT_PT_IND with 'Unknown'
final_df['FT_PT_IND'] = final_df['FT_PT_IND'].fillna('Unknown')

# Convert TransactionClass to numeric values
label_encoder = LabelEncoder()
final_df['TransactionClass'] = label_encoder.fit_transform(final_df['TransactionClass'])

# One-hot encode categorical variables
final_df_encoded = pd.get_dummies(final_df, columns=['GENDER_CODE', 'Residency', 'FT_PT_IND', 'COLL_DESC'])

# Drop the unnecessary columns
final_df_encoded = final_df_encoded.drop(columns=['STUDENT_ID_NUMBER', 'TransactionCount'])

# Split the data into features (X) and target (y)
X = final_df_encoded.drop(columns=['TransactionClass'])
y = final_df_encoded['TransactionClass']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Initialize the Logistic Regression model with class weights
logreg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

# Train the model
logreg.fit(X_train, y_train)

# Make predictions
logreg_preds = logreg.predict(X_test)

# Get the class names in the correct order
class_names = label_encoder.inverse_transform([0, 1, 2])

# Evaluate the model
print("Logistic Regression Classification Report")
print(classification_report(y_test, logreg_preds, target_names=class_names))
print("Accuracy:", accuracy_score(y_test, logreg_preds))
print(confusion_matrix(y_test, logreg_preds))


# In[312]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Train the model
rf.fit(X_train, y_train)

# Make predictions
rf_preds = rf.predict(X_test)

# Evaluate the model
print("\nRandom Forest Classification Report")
print(classification_report(y_test, rf_preds, target_names=class_names))
print("Accuracy:", accuracy_score(y_test, rf_preds))
print(confusion_matrix(y_test, rf_preds))


# In[313]:


from sklearn.tree import DecisionTreeClassifier

# Initialize the Decision Tree model
dtree = DecisionTreeClassifier(random_state=42, class_weight='balanced')

# Train the model
dtree.fit(X_train, y_train)

# Make predictions
dtree_preds = dtree.predict(X_test)

# Evaluate the model
print("\nDecision Tree Classification Report")
print(classification_report(y_test, dtree_preds, target_names=class_names))
print("Accuracy:", accuracy_score(y_test, dtree_preds))
print(confusion_matrix(y_test, dtree_preds))


# ## Feature Engineering

# In[314]:


import pandas as pd

# Combine the two membership dataframes
#combined_membership = pd.concat([membership_2223, membership_2324]).drop_duplicates()

# Ensure CardID and STUDENT_ID_NUMBER are strings for consistency
eats_reg['CardID'] = eats_reg['CardID'].astype(str)
eats_reg['STUDENT_ID_NUMBER'] = eats_reg['STUDENT_ID_NUMBER'].astype(str)
combined_membership['STUDENT_ID_NUMBER'] = combined_membership['STUDENT_ID_NUMBER'].astype(str)

# Merge combined_membership with eats registration data
students_with_eats_all = combined_membership.merge(eats_reg[['CardID', 'STUDENT_ID_NUMBER']], on='STUDENT_ID_NUMBER', how='inner')

# Aggregate the number of transactions per student
transactions_per_student = eats_txns.groupby('CardID').size().reset_index(name='TransactionCount')

# Merge with transactions data to get total transactions per student
students_with_eats_all = students_with_eats_all.merge(transactions_per_student[['CardID', 'TransactionCount']], on='CardID', how='left')

# Fill NaN values in TransactionCount with 0 (students who registered but made no transactions)
students_with_eats_all['TransactionCount'] = students_with_eats_all['TransactionCount'].fillna(0)

# Calculate the quartiles
quartiles = students_with_eats_all['TransactionCount'].quantile([0.25, 0.5, 0.75])

# Define the classification function based on quartiles
def classify_transaction_count(transaction_count):
    if transaction_count <= quartiles[0.25]:
        return 'Low'
    elif transaction_count <= quartiles[0.75]:
        return 'Medium'
    else:
        return 'High'

# Apply the classification
students_with_eats_all['TransactionClass'] = students_with_eats_all['TransactionCount'].apply(classify_transaction_count)

# Final DataFrame with all columns
final_df_test = students_with_eats_all

# Verify the DataFrame
print(final_df_test.columns)
print(final_df_test.head())


# In[315]:


import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot stacked bar charts
def plot_stacked_bar(df, feature, target='TransactionClass'):
    crosstab = pd.crosstab(df[feature], df[target], normalize='index')
    crosstab.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis')
    plt.title(f'Stacked Bar Chart of {feature} vs {target}')
    plt.xlabel(feature)
    plt.ylabel('Proportion')
    plt.legend(title=target, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# List of features to plot
features_to_plot = ['GENDER_CODE', 'RESIDENCY_DESC', 'YEAR_CODE', 'FT_PT_IND', 'COLL_DESC', 'NATIONALITY_DESC', 'PROGRAMME_DESC', 'STUDENT_LEVEL', 'LEVEL_DESC', 'RELIGION_DESC', 'ETHNIC_DESC', 'RESIDENCY_CODE', 'SEXORT_DESC', 'DISABILITY_INFO']

# Plot stacked bar charts for each feature
for feature in features_to_plot:
    plot_stacked_bar(final_df_test, feature)


# ## Classification Based on Engagement

# In[316]:


all_events_data_new.head()


# In[317]:


# Count the number of events each student participated in
event_participation_count = all_events_data_new['Student ID'].value_counts().reset_index()
event_participation_count.columns = ['STUDENT_ID_NUMBER', 'EventParticipationCount']

# Verify the new dataframe
print("Event Participation Count DataFrame:")
print(event_participation_count.head())

# Merge this data with the combined membership data
engage_df = combined_membership.merge(event_participation_count, on='STUDENT_ID_NUMBER', how='inner')

# Fill NaN values in EventParticipationCount with 0 (students who did not participate in any event)
engage_df['EventParticipationCount'] = engage_df['EventParticipationCount'].fillna(0)

# Verify the new dataframe
print("Engage DataFrame:")
print(engage_df.head())


# In[353]:


engage_df.to_csv('/Users/pranshu/Desktop/DissData/engage.csv')


# In[318]:


all_events_data_new['Event Date'].min()


# In[319]:


eats_txns['TransactionDate'].min()


# In[320]:


# Calculate the quartiles for EventParticipationCount
engagement_quartiles = engage_df['EventParticipationCount'].quantile([0.25, 0.5, 0.75])

# Define the classification function based on engagement quartiles
def classify_engagement_count(participation_count):
    if participation_count <= engagement_quartiles[0.25]:
        return 'Low'
    elif participation_count <= engagement_quartiles[0.75]:
        return 'Medium'
    else:
        return 'High'

# Apply the classification
engage_df['EngagementClass'] = engage_df['EventParticipationCount'].apply(classify_engagement_count)

# Select the relevant columns for classification modeling
final_engage_df = engage_df[['STUDENT_ID_NUMBER', 'GENDER_CODE', 'DATE_OF_BIRTH', 'RESIDENCY_DESC', 'YEAR_CODE', 'FT_PT_IND', 'COLL_DESC', 'EventParticipationCount', 'EngagementClass']]

# Verify the final dataframe
print("Final Engage DataFrame:")
print(final_engage_df.head())


# In[321]:


engagement_quartiles


# In[322]:


import matplotlib.pyplot as plt

# Plot a histogram of EventParticipationCount
plt.figure(figsize=(12, 7))
plt.hist(engage_df['EventParticipationCount'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Event Participation Count')
plt.xlabel('Number of Events Participated')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[323]:


combined_membership['STUDENT_ID_NUMBER'].nunique()


# In[324]:


engage_df['STUDENT_ID_NUMBER'].nunique()


# In[325]:


import matplotlib.pyplot as plt

# Plot a histogram of EventParticipationCount
plt.figure(figsize=(12, 7))
plt.hist(engage_df['EventParticipationCount'], bins=30, color='skyblue', edgecolor='black')

# Add lines for the quantiles
quantiles = engage_df['EventParticipationCount'].quantile([0.25, 0.5, 0.75])
for quantile in quantiles:
    plt.axvline(quantile, color='r', linestyle='dashed', linewidth=2)

plt.title('Distribution of Event Participation Count with Quartiles')
plt.xlabel('Number of Events Participated')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[326]:


# Calculate the number of students in each engagement class
engagement_class_counts = final_engage_df['EngagementClass'].value_counts()

# Print the counts
print("Number of students in each engagement class:")
print(engagement_class_counts)


# In[277]:


final_engage_df['STUDENT_ID_NUMBER'].nunique()


# In[392]:


# Find duplicate student IDs
duplicate_students = final_engage_df[final_engage_df.duplicated('STUDENT_ID_NUMBER', keep=False)]

# Print the rows with duplicate student IDs
print("Rows with duplicate student IDs:")
print(duplicate_students)


# In[328]:


# Find duplicate student IDs
duplicate_students = final_engage_df[final_engage_df.duplicated('STUDENT_ID_NUMBER', keep=False)]

# Check if all duplicates have the same values across all columns
duplicates_same_values = duplicate_students.groupby('STUDENT_ID_NUMBER').nunique()

# Filter for rows where all columns have only one unique value, indicating exact duplicates
exact_duplicates = duplicates_same_values[(duplicates_same_values == 1).all(axis=1)]

# Print the rows with exact duplicates
print("Rows with exact duplicate values:")
print(final_engage_df[final_engage_df['STUDENT_ID_NUMBER'].isin(exact_duplicates.index)])


# In[329]:


# Find duplicate student IDs
duplicate_students = event_participation_count[event_participation_count.duplicated('STUDENT_ID_NUMBER', keep=False)]

# Print the rows with duplicate student IDs
print("Rows with duplicate student IDs:")
print(duplicate_students)


# In[330]:


# Find duplicate student IDs
duplicate_students = engage_df[engage_df.duplicated('STUDENT_ID_NUMBER', keep=False)]

# Print the rows with duplicate student IDs
print("Rows with duplicate student IDs:")
print(duplicate_students)


# In[331]:


# Find duplicate student IDs
duplicate_students = combined_membership[combined_membership.duplicated('STUDENT_ID_NUMBER', keep=False)]

# Print the rows with duplicate student IDs
print("Rows with duplicate student IDs:")
print(duplicate_students)


# In[333]:


combined_membership[combined_membership['STUDENT_ID_NUMBER'] == '201083055']


# In[334]:


combined_membership[combined_membership['STUDENT_ID_NUMBER'] == '201669458']


# In[335]:


import matplotlib.pyplot as plt
import seaborn as sns

# Define the features to plot
features = ['GENDER_CODE', 'DATE_OF_BIRTH', 'RESIDENCY_DESC', 'YEAR_CODE', 'FT_PT_IND', 'COLL_DESC']

# Set the plot style
sns.set(style="whitegrid")

# Plot each feature
for feature in features:
    # Create a crosstab of EngagementClass and the feature
    crosstab = pd.crosstab(final_engage_df[feature], final_engage_df['EngagementClass'], normalize='index') * 100
    
    # Plot the crosstab as a stacked bar chart
    crosstab.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis')
    
    # Customize the plot
    plt.title(f'Engagement Class Distribution by {feature}')
    plt.xlabel(feature)
    plt.ylabel('Percentage')
    plt.legend(title='Engagement Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Show the plot
    plt.show()


# In[395]:


import pandas as pd
import matplotlib.pyplot as plt

# Function to plot stacked 100% column chart
def plot_stacked_bar(ax, df, feature, title):
    crosstab = pd.crosstab(df[feature], df['EngagementClass'], normalize='index') * 100
    crosstab.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    ax.set_title(title)
    ax.set_xlabel(feature)
    ax.set_ylabel('Percentage')
    ax.legend(loc='upper right', title='Engagement Class')

# Number of features
features = ['GENDER_CODE', 'Age', 'Residency', 'YEAR_CODE', 'COLL_DESC']
n_features = len(features)

# Determine grid size
n_cols = 2  # Number of columns
n_rows = (n_features + 1) // n_cols  # Number of rows (ensure all features fit in the grid)

# Create subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
axs = axs.flatten()  # Flatten to make indexing easier

# Plot each feature in a grid
for i, feature in enumerate(features):
    plot_stacked_bar(axs[i], final_engage_df, feature, f'Distribution of Engagement Classes by {feature}')

# Remove any unused subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()


# In[336]:


import pandas as pd

# Define the features to analyze
features = ['GENDER_CODE', 'DATE_OF_BIRTH', 'RESIDENCY_DESC', 'YEAR_CODE', 'FT_PT_IND', 'COLL_DESC']

# Initialize a dictionary to store crosstab results
crosstab_results = {}

# Create a crosstab for each feature
for feature in features:
    crosstab = pd.crosstab(final_engage_df[feature], final_engage_df['EngagementClass'], normalize='index') * 100
    crosstab_results[feature] = crosstab

# Display the crosstab results
for feature, crosstab in crosstab_results.items():
    print(f'Crosstab for {feature}:')
    print(crosstab)
    print('\n')


# In[337]:


import pandas as pd

# Convert birth year to age
current_year = pd.to_datetime('today').year
final_engage_df['Age'] = current_year - final_engage_df['DATE_OF_BIRTH']

# Categorize residency as Home or International
def classify_residency(residency):
    if 'Home' in residency:
        return 'Home'
    else:
        return 'International'

final_engage_df['Residency'] = final_engage_df['RESIDENCY_DESC'].apply(classify_residency)

# Drop the original 'DATE_OF_BIRTH' and 'RESIDENCY_DESC' columns
final_engage_df = final_engage_df.drop(columns=['DATE_OF_BIRTH', 'RESIDENCY_DESC'])

# Verify the changes
print(final_engage_df.head())


# In[339]:


import pandas as pd

# Define the mapping function for YEAR_CODE
def map_year_code(year_code):
    if 'A' in year_code:
        return int(year_code[0])  # Convert '2A' to 2, '3A' to 3, etc.
    elif len(year_code) == 2 and year_code[0] == '0':
        return int(year_code[1])  # Convert '01' to 1, '02' to 2, etc.
    else:
        return int(year_code)  # Convert numeric strings directly to integers

# Apply the mapping function to the YEAR_CODE column
final_engage_df['YEAR_CODE'] = final_engage_df['YEAR_CODE'].apply(map_year_code)

# Verify the changes
print(final_engage_df['YEAR_CODE'].unique())
print(final_engage_df.head())


# In[340]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Drop 'EventParticipationCount' and 'STUDENT_ID_NUMBER'
X = final_engage_df.drop(columns=['EventParticipationCount', 'STUDENT_ID_NUMBER', 'EngagementClass'])
y = final_engage_df['EngagementClass']

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=['GENDER_CODE', 'Residency', 'FT_PT_IND', 'COLL_DESC'])

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Classification Report")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# XGBoost Classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("XGBoost Classification Report")
print(classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))


# In[343]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42,class_weight='balanced')

# Initialize the GridSearchCV object
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)

# Fit the grid search to the data
grid_search_rf.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", grid_search_rf.best_params_)

# Predict using the best model
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# Evaluate the model
print("Random Forest Classification Report")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))


# In[ ]:




