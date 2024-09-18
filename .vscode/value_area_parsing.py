import pandas as pd
import re
import os

def parse_file(file_path):
    # Get the file name
    file_name = os.path.basename(file_path)
    print(file_name)

    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into sections for each date
    date_sections = re.split(r'Date\s*-->\s*', content)[1:]  # Skip the first empty split

    data_list = []
    for section in date_sections:
        lines = section.strip().split('\n')
        date = lines[0].strip()
        
        # Use regex to extract key-value pairs
        pattern = r'(.*?)\s*-->\s*(.*)'
        matches = re.findall(pattern, '\n'.join(lines[1:]))

        # Create a dictionary from the matches
        data = {'Date': date}
        data.update({key.strip(): value.strip() for key, value in matches})
        data_list.append(data)

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data_list)    
    df['Stock_Name'] = file_name.split('.')[0]
    
    return df

# Specify the folder path
folder_path = 'E:/Amibroker/Historical Data with Value Area Levels'

# Get a list of all .txt files in the folder
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# Create an empty DataFrame to store all the data
all_data = pd.DataFrame()

# Iterate through each .txt file
for file_name in txt_files:
    file_path = os.path.join(folder_path, file_name)
    df = parse_file(file_path)
    all_data = pd.concat([all_data, df], ignore_index=True)

# Save the result to a CSV file
all_data.to_csv('./data/value_area_levels_for_all_stocks_new.csv', index=False)
print("Parsing complete. Data saved to 'parsed_value_area_data.csv'")
