# Exploring my Running Data to Find my Most Grueling Run Ever (Based On Numbers of Course)
Here I will be exploring my college running data from 2017 to 2022. I'm doing this as a sort of "last horah" now that I am offically a washed up runner with a cycling and snowboarding problem.


```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import statistics
```

    Mounted at /content/drive



```python
data_path = "/content/drive/MyDrive/Running Exploration/Activities 20"
df = pd.read_csv(data_path + "17.csv")
for i in range (18,23):
  df = pd.concat([df, pd.read_csv(data_path + str(i) + ".csv")])
df = df.reset_index()
```


```python
df.head().to_markdown()
```




    '|    |   index | Activity Type   | Date                | Favorite   | Title               |   Distance | Calories   | Time     |   Avg HR |   Max HR |   Avg Run Cadence |   Max Run Cadence | Avg Pace   | Best Pace   |   Total Ascent |   Total Descent |   Avg Stride Length |   Avg Vertical Ratio |   Avg Vertical Oscillation |   Avg Ground Contact Time |   Training Stress Score® |   Avg Power |   Max Power |   Grit |   Flow |   Avg. Swolf |   Avg Stroke Rate |   Total Reps | Dive Time   |   Min Temp | Surface Interval   | Decompression   | Best Lap Time   |   Number of Laps |   Max Temp | Moving Time   | Elapsed Time   | Min Elevation   | Max Elevation   |\n|---:|--------:|:----------------|:--------------------|:-----------|:--------------------|-----------:|:-----------|:---------|---------:|---------:|------------------:|------------------:|:-----------|:------------|---------------:|----------------:|--------------------:|---------------------:|---------------------------:|--------------------------:|-------------------------:|------------:|------------:|-------:|-------:|-------------:|------------------:|-------------:|:------------|-----------:|:-------------------|:----------------|:----------------|-----------------:|-----------:|:--------------|:---------------|:----------------|:----------------|\n|  0 |       0 | Running         | 2017-12-31 10:48:59 | False      | Sevierville Running |      13.62 | 1,462      | 01:29:31 |        0 |        0 |               175 |               187 | 6:34       | 5:59        |            381 |             425 |                1.4  |                    0 |                          0 |                         0 |                        0 |           0 |           0 |      0 |      0 |            0 |                 0 |            0 | 0:00        |          0 | 0:00               | No              | 00:00.00        |               14 |          0 | 01:29:30      | 01:32:18       | 958             | 1,181           |\n|  1 |       1 | Running         | 2017-12-30 08:44:33 | False      | Sevierville Running |       5.83 | 631        | 00:41:23 |        0 |        0 |               176 |               191 | 7:06       | 6:32        |            169 |               9 |                1.29 |                    0 |                          0 |                         0 |                        0 |           0 |           0 |      0 |      0 |            0 |                 0 |            0 | 0:00        |          0 | 0:00               | No              | 00:00.00        |                6 |          0 | 00:41:25      | 00:41:35       | 1,001           | 1,174           |\n|  2 |       2 | Running         | 2017-12-29 11:21:54 | False      | Sevierville Running |       8.22 | 881        | 00:50:45 |        0 |        0 |               176 |               191 | 6:10       | 5:21        |            285 |             184 |                1.47 |                    0 |                          0 |                         0 |                        0 |           0 |           0 |      0 |      0 |            0 |                 0 |            0 | 0:00        |          0 | 0:00               | No              | 00:00.00        |               10 |          0 | 00:50:45      | 00:51:27       | 968             | 1,167           |\n|  3 |       3 | Running         | 2017-12-29 11:06:10 | False      | Sevierville Running |       1.97 | 209        | 00:13:42 |        0 |        0 |               174 |               191 | 6:57       | 6:19        |             48 |             181 |                1.34 |                    0 |                          0 |                         0 |                        0 |           0 |           0 |      0 |      0 |            0 |                 0 |            0 | 0:00        |          0 | 0:00               | No              | 00:00.00        |                2 |          0 | 00:13:41      | 00:14:01       | 1,028           | 1,178           |\n|  4 |       4 | Running         | 2017-12-28 06:24:02 | False      | Moss Point Running  |       7.37 | 797        | 00:52:04 |        0 |        0 |               173 |               185 | 7:04       | 6:23        |            182 |             195 |                1.32 |                    0 |                          0 |                         0 |                        0 |           0 |           0 |      0 |      0 |            0 |                 0 |            0 | 0:00        |          0 | 0:00               | No              | 00:00.00        |                8 |          0 | 00:52:03      | 00:52:16       | 64              | 135             |'



## Feature Engineering

Let's have a look at the columns we have in our dataset and decide which ones may be useful to explore.


```python
df.columns
```




    Index(['index', 'Activity Type', 'Date', 'Favorite', 'Title', 'Distance',
           'Calories', 'Time', 'Avg HR', 'Max HR', 'Avg Run Cadence',
           'Max Run Cadence', 'Avg Pace', 'Best Pace', 'Total Ascent',
           'Total Descent', 'Avg Stride Length', 'Avg Vertical Ratio',
           'Avg Vertical Oscillation', 'Avg Ground Contact Time',
           'Training Stress Score®', 'Avg Power', 'Max Power', 'Grit', 'Flow',
           'Avg. Swolf', 'Avg Stroke Rate', 'Total Reps', 'Dive Time', 'Min Temp',
           'Surface Interval', 'Decompression', 'Best Lap Time', 'Number of Laps',
           'Max Temp', 'Moving Time', 'Elapsed Time', 'Min Elevation',
           'Max Elevation'],
          dtype='object')




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1975 entries, 0 to 1974
    Data columns (total 39 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   index                     1975 non-null   int64  
     1   Activity Type             1975 non-null   object 
     2   Date                      1975 non-null   object 
     3   Favorite                  1975 non-null   bool   
     4   Title                     1975 non-null   object 
     5   Distance                  1975 non-null   float64
     6   Calories                  1975 non-null   object 
     7   Time                      1975 non-null   object 
     8   Avg HR                    1975 non-null   int64  
     9   Max HR                    1975 non-null   int64  
     10  Avg Run Cadence           1975 non-null   object 
     11  Max Run Cadence           1975 non-null   object 
     12  Avg Pace                  1975 non-null   object 
     13  Best Pace                 1975 non-null   object 
     14  Total Ascent              1975 non-null   object 
     15  Total Descent             1975 non-null   object 
     16  Avg Stride Length         1975 non-null   float64
     17  Avg Vertical Ratio        1975 non-null   float64
     18  Avg Vertical Oscillation  1975 non-null   float64
     19  Avg Ground Contact Time   1975 non-null   int64  
     20  Training Stress Score®    1975 non-null   float64
     21  Avg Power                 1975 non-null   int64  
     22  Max Power                 1975 non-null   int64  
     23  Grit                      1975 non-null   float64
     24  Flow                      1975 non-null   float64
     25  Avg. Swolf                1975 non-null   int64  
     26  Avg Stroke Rate           1975 non-null   int64  
     27  Total Reps                1975 non-null   int64  
     28  Dive Time                 1975 non-null   object 
     29  Min Temp                  1975 non-null   float64
     30  Surface Interval          1975 non-null   object 
     31  Decompression             1975 non-null   object 
     32  Best Lap Time             1975 non-null   object 
     33  Number of Laps            1975 non-null   object 
     34  Max Temp                  1975 non-null   float64
     35  Moving Time               1975 non-null   object 
     36  Elapsed Time              1975 non-null   object 
     37  Min Elevation             1975 non-null   object 
     38  Max Elevation             1975 non-null   object 
    dtypes: bool(1), float64(9), int64(9), object(20)
    memory usage: 588.4+ KB


To give a good understanding of my running trends and eventually find the most grueling training week of my career, I want to focus on the distance, speed, elevation gain/loss, heart rate, cadence, and tempurature of my runs. I admitedly have quite a bit of domain knowledge when it comes to running, and these are oftentimes the most crucial stats for quantifying running performances. Let's clean up the columns contatining this data and drop the ones we don't need.


```python
cols = ['Date', 'Title', 'Distance',
       'Calories', 'Time', 'Avg HR', 'Max HR', 'Avg Run Cadence',
       'Max Run Cadence', 'Avg Pace', 'Best Pace', 'Total Ascent',
       'Total Descent', 'Avg Stride Length', 'Min Temp', 'Max Temp', 'Min Elevation',
       'Max Elevation']
cols_to_check = ['Calories', 'Total Ascent', 'Total Descent', 'Min Elevation', 'Max Elevation']
df = df[cols]
df[cols_to_check] = df[cols_to_check].replace({',':''}, regex=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'] < np.datetime64("2022-05-15")]
df = df.replace('--', None)
df = df.replace(0.0, None)
df = df.replace(0, None)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1890 entries, 0 to 1974
    Data columns (total 18 columns):
     #   Column             Non-Null Count  Dtype         
    ---  ------             --------------  -----         
     0   Date               1890 non-null   datetime64[ns]
     1   Title              1890 non-null   object        
     2   Distance           1889 non-null   object        
     3   Calories           1890 non-null   object        
     4   Time               1890 non-null   object        
     5   Avg HR             608 non-null    object        
     6   Max HR             608 non-null    object        
     7   Avg Run Cadence    1887 non-null   object        
     8   Max Run Cadence    1887 non-null   object        
     9   Avg Pace           1886 non-null   object        
     10  Best Pace          1886 non-null   object        
     11  Total Ascent       1855 non-null   object        
     12  Total Descent      1861 non-null   object        
     13  Avg Stride Length  1886 non-null   object        
     14  Min Temp           608 non-null    object        
     15  Max Temp           608 non-null    object        
     16  Min Elevation      1879 non-null   object        
     17  Max Elevation      1884 non-null   object        
    dtypes: datetime64[ns](1), object(17)
    memory usage: 280.5+ KB



```python
df = df.dropna()
```


```python
df['Distance'] = df['Distance'].astype(float)
df['Calories'] = df['Calories'].astype(float)

df['Avg HR'] = df['Avg HR'].astype(int)
df['Max HR'] = df['Max HR'].astype(int)
df['Avg Run Cadence'] = df['Avg Run Cadence'].astype(int)
df['Max Run Cadence'] = df['Max Run Cadence'].astype(int)
df['Min Temp'] = df['Min Temp'].astype(int)
df['Max Temp'] = df['Max Temp'].astype(int)
df['Min Elevation'] = df['Min Elevation'].astype(int)
df['Max Elevation'] = df['Max Elevation'].astype(int)
df['Total Ascent'] = df['Total Ascent'].astype(int)
df['Total Descent'] = df['Total Descent'].astype(int)
df['Avg Stride Length'] = df['Avg Stride Length'].astype(float)




df['Total Run Time Mins'] = [60 * int(x.split(':')[0]) + int(x.split(':')[1]) + (int(x.split(':')[2].split('.')[0])/60) for x in df['Time']]
df['Avg Pace Mins'] = [int(x.split(':')[0]) + int(x.split(':')[1]) / 60 for x in df['Avg Pace']]
df['Best Pace Mins'] = [int(x.split(':')[0]) + int(x.split(':')[1]) / 60 for x in df['Best Pace']]
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 602 entries, 963 to 1974
    Data columns (total 21 columns):
     #   Column               Non-Null Count  Dtype         
    ---  ------               --------------  -----         
     0   Date                 602 non-null    datetime64[ns]
     1   Title                602 non-null    object        
     2   Distance             602 non-null    float64       
     3   Calories             602 non-null    float64       
     4   Time                 602 non-null    object        
     5   Avg HR               602 non-null    int64         
     6   Max HR               602 non-null    int64         
     7   Avg Run Cadence      602 non-null    int64         
     8   Max Run Cadence      602 non-null    int64         
     9   Avg Pace             602 non-null    object        
     10  Best Pace            602 non-null    object        
     11  Total Ascent         602 non-null    int64         
     12  Total Descent        602 non-null    int64         
     13  Avg Stride Length    602 non-null    float64       
     14  Min Temp             602 non-null    int64         
     15  Max Temp             602 non-null    int64         
     16  Min Elevation        602 non-null    int64         
     17  Max Elevation        602 non-null    int64         
     18  Total Run Time Mins  602 non-null    float64       
     19  Avg Pace Mins        602 non-null    float64       
     20  Best Pace Mins       602 non-null    float64       
    dtypes: datetime64[ns](1), float64(6), int64(10), object(4)
    memory usage: 103.5+ KB



```python
df = df.drop(columns = ['Avg Pace', 'Best Pace', 'Time'])
```


```python
df
```





  <div id="df-df70ee61-4704-42ca-ae6c-b2bdcfb4cd71" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Title</th>
      <th>Distance</th>
      <th>Calories</th>
      <th>Avg HR</th>
      <th>Max HR</th>
      <th>Avg Run Cadence</th>
      <th>Max Run Cadence</th>
      <th>Total Ascent</th>
      <th>Total Descent</th>
      <th>Avg Stride Length</th>
      <th>Min Temp</th>
      <th>Max Temp</th>
      <th>Min Elevation</th>
      <th>Max Elevation</th>
      <th>Total Run Time Mins</th>
      <th>Avg Pace Mins</th>
      <th>Best Pace Mins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>963</th>
      <td>2020-12-31 09:15:19</td>
      <td>Jackson County Running</td>
      <td>10.00</td>
      <td>843.0</td>
      <td>141</td>
      <td>155</td>
      <td>175</td>
      <td>184</td>
      <td>108</td>
      <td>112</td>
      <td>1.31</td>
      <td>69</td>
      <td>77</td>
      <td>-23</td>
      <td>-3</td>
      <td>70.016667</td>
      <td>7.000000</td>
      <td>6.366667</td>
    </tr>
    <tr>
      <th>964</th>
      <td>2020-12-30 09:13:33</td>
      <td>Mobile County Running</td>
      <td>12.00</td>
      <td>1044.0</td>
      <td>147</td>
      <td>165</td>
      <td>174</td>
      <td>184</td>
      <td>200</td>
      <td>194</td>
      <td>1.40</td>
      <td>69</td>
      <td>80</td>
      <td>155</td>
      <td>206</td>
      <td>79.383333</td>
      <td>6.616667</td>
      <td>5.666667</td>
    </tr>
    <tr>
      <th>965</th>
      <td>2020-12-29 09:23:17</td>
      <td>Jackson County Running</td>
      <td>11.16</td>
      <td>865.0</td>
      <td>146</td>
      <td>184</td>
      <td>175</td>
      <td>196</td>
      <td>154</td>
      <td>164</td>
      <td>1.47</td>
      <td>66</td>
      <td>82</td>
      <td>77</td>
      <td>116</td>
      <td>69.566667</td>
      <td>6.233333</td>
      <td>4.450000</td>
    </tr>
    <tr>
      <th>966</th>
      <td>2020-12-28 09:33:42</td>
      <td>Mobile County Running</td>
      <td>9.08</td>
      <td>698.0</td>
      <td>136</td>
      <td>150</td>
      <td>173</td>
      <td>194</td>
      <td>154</td>
      <td>151</td>
      <td>1.33</td>
      <td>68</td>
      <td>82</td>
      <td>11</td>
      <td>76</td>
      <td>63.516667</td>
      <td>7.000000</td>
      <td>6.150000</td>
    </tr>
    <tr>
      <th>967</th>
      <td>2020-12-27 09:20:00</td>
      <td>Mobile County Running</td>
      <td>14.01</td>
      <td>1166.0</td>
      <td>147</td>
      <td>167</td>
      <td>176</td>
      <td>188</td>
      <td>558</td>
      <td>591</td>
      <td>1.41</td>
      <td>62</td>
      <td>77</td>
      <td>98</td>
      <td>284</td>
      <td>91.033333</td>
      <td>6.500000</td>
      <td>5.533333</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>2022-01-05 08:38:41</td>
      <td>Starkville Running</td>
      <td>11.50</td>
      <td>1014.0</td>
      <td>147</td>
      <td>163</td>
      <td>178</td>
      <td>201</td>
      <td>420</td>
      <td>404</td>
      <td>1.30</td>
      <td>57</td>
      <td>75</td>
      <td>223</td>
      <td>402</td>
      <td>79.966667</td>
      <td>6.950000</td>
      <td>6.316667</td>
    </tr>
    <tr>
      <th>1971</th>
      <td>2022-01-04 10:30:42</td>
      <td>Noxubee County Running</td>
      <td>11.01</td>
      <td>851.0</td>
      <td>145</td>
      <td>169</td>
      <td>183</td>
      <td>232</td>
      <td>289</td>
      <td>246</td>
      <td>1.42</td>
      <td>51</td>
      <td>69</td>
      <td>130</td>
      <td>201</td>
      <td>67.800000</td>
      <td>6.166667</td>
      <td>4.483333</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>2022-01-03 10:29:42</td>
      <td>Starkville Running</td>
      <td>15.01</td>
      <td>1349.0</td>
      <td>148</td>
      <td>171</td>
      <td>179</td>
      <td>190</td>
      <td>807</td>
      <td>774</td>
      <td>1.31</td>
      <td>37</td>
      <td>77</td>
      <td>89</td>
      <td>243</td>
      <td>103.116667</td>
      <td>6.866667</td>
      <td>5.650000</td>
    </tr>
    <tr>
      <th>1973</th>
      <td>2022-01-02 09:07:55</td>
      <td>Leon County Running</td>
      <td>8.01</td>
      <td>688.0</td>
      <td>141</td>
      <td>156</td>
      <td>177</td>
      <td>188</td>
      <td>810</td>
      <td>978</td>
      <td>1.29</td>
      <td>77</td>
      <td>86</td>
      <td>-94</td>
      <td>167</td>
      <td>56.416667</td>
      <td>7.050000</td>
      <td>5.716667</td>
    </tr>
    <tr>
      <th>1974</th>
      <td>2022-01-01 09:45:26</td>
      <td>Tallahassee Running</td>
      <td>7.00</td>
      <td>593.0</td>
      <td>136</td>
      <td>159</td>
      <td>175</td>
      <td>186</td>
      <td>801</td>
      <td>863</td>
      <td>1.26</td>
      <td>75</td>
      <td>84</td>
      <td>-59</td>
      <td>205</td>
      <td>51.183333</td>
      <td>7.316667</td>
      <td>5.300000</td>
    </tr>
  </tbody>
</table>
<p>602 rows × 18 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-df70ee61-4704-42ca-ae6c-b2bdcfb4cd71')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-df70ee61-4704-42ca-ae6c-b2bdcfb4cd71 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-df70ee61-4704-42ca-ae6c-b2bdcfb4cd71');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-eea24254-61c5-4ed5-9ae7-39489205274d">
  <button class="colab-df-quickchart" onclick="quickchart('df-eea24254-61c5-4ed5-9ae7-39489205274d')"
            title="Suggest charts."
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
    background-color: #E8F0FE;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: #1967D2;
    height: 32px;
    padding: 0 0 0 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: #E2EBFA;
    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: #174EA6;
  }

  [theme=dark] .colab-df-quickchart {
    background-color: #3B4455;
    fill: #D2E3FC;
  }

  [theme=dark] .colab-df-quickchart:hover {
    background-color: #434B5C;
    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
    fill: #FFFFFF;
  }
</style>

  <script>
    async function quickchart(key) {
      const charts = await google.colab.kernel.invokeFunction(
          'suggestCharts', [key], {});
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-eea24254-61c5-4ed5-9ae7-39489205274d button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_f8be78a4-a8f3-4188-99a7-596274920641">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_f8be78a4-a8f3-4188-99a7-596274920641 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 602 entries, 963 to 1974
    Data columns (total 18 columns):
     #   Column               Non-Null Count  Dtype         
    ---  ------               --------------  -----         
     0   Date                 602 non-null    datetime64[ns]
     1   Title                602 non-null    object        
     2   Distance             602 non-null    float64       
     3   Calories             602 non-null    float64       
     4   Avg HR               602 non-null    int64         
     5   Max HR               602 non-null    int64         
     6   Avg Run Cadence      602 non-null    int64         
     7   Max Run Cadence      602 non-null    int64         
     8   Total Ascent         602 non-null    int64         
     9   Total Descent        602 non-null    int64         
     10  Avg Stride Length    602 non-null    float64       
     11  Min Temp             602 non-null    int64         
     12  Max Temp             602 non-null    int64         
     13  Min Elevation        602 non-null    int64         
     14  Max Elevation        602 non-null    int64         
     15  Total Run Time Mins  602 non-null    float64       
     16  Avg Pace Mins        602 non-null    float64       
     17  Best Pace Mins       602 non-null    float64       
    dtypes: datetime64[ns](1), float64(6), int64(10), object(1)
    memory usage: 89.4+ KB



```python
from sklearn.cluster import KMeans
import numpy as np

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(df[['Max HR', 'Best Pace Mins']])
```


```python
Counter(kmeans.predict(df[['Max HR', 'Best Pace Mins']]))
```




    Counter({0: 215, 1: 387})




```python
import numpy as np
X = df[['Distance', 'Calories',
       'Avg Run Cadence', 'Max Run Cadence', 'Total Ascent', 'Total Descent',
       'Avg Stride Length', 'Min Temp', 'Max Temp', 'Total Run Time Mins', 'Avg Pace Mins',
       'Best Pace Mins']].to_numpy()
y = df[['Avg HR']].to_numpy()
```


```python
from sklearn.feature_selection import SelectKBest, chi2
X_new = SelectKBest(chi2, k=8).fit_transform(X, y)
X_new.shape
```




    (602, 8)




```python
from sklearn.metrics import mean_squared_error
```


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
```


```python
from sklearn import linear_model
lasso = linear_model.Lasso(alpha = 0.1)
lasso.fit(X, y)
p = lasso.predict(X)
mean_squared_error(list(df['Avg HR']), p)
```




    19.278796045486203




```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, y)
p = reg.predict(X)
mean_squared_error(list(df['Avg HR']), p)
```




    18.66823958606139




```python
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X, y)
p = reg.predict(X)
mean_squared_error(list(df['Avg HR']), p)
```

    /usr/local/lib/python3.10/dist-packages/sklearn/utils/validation.py:1143: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)





    18.66823958606139




```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
rfr = RandomForestRegressor(max_depth=10)
rfr.fit(X, y)
p = reg.predict(X)
mean_squared_error(list(df['Avg HR']), p)
```

    <ipython-input-23-12a0ef341382>:4: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rfr.fit(X, y)





    18.66823958606139




```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
import statistics
scores = cross_validate(lasso, X, y, cv=5, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)
statistics.mean(scores['test_neg_mean_squared_error'])
```




    -21.322112780383843




```python
scores = cross_validate(reg, X, y, cv=5, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)
statistics.mean(scores['test_neg_mean_squared_error'])
```




    -22.80959070422728




```python
scores = cross_validate(regr, X, y, cv=5, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)
statistics.mean(scores['test_neg_mean_squared_error'])
```

    /usr/local/lib/python3.10/dist-packages/sklearn/utils/validation.py:1143: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    /usr/local/lib/python3.10/dist-packages/sklearn/utils/validation.py:1143: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    /usr/local/lib/python3.10/dist-packages/sklearn/utils/validation.py:1143: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    /usr/local/lib/python3.10/dist-packages/sklearn/utils/validation.py:1143: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    /usr/local/lib/python3.10/dist-packages/sklearn/utils/validation.py:1143: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)





    -60.11779594114683




```python
scores = cross_validate(rfr, X, y, cv=5, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)
statistics.mean(scores['test_neg_mean_squared_error'])
```

    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_validation.py:686: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)





    -29.10460008095694




```python

```


```python
import numpy as np
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=2, random_state=0).fit(df[['Best Pace Mins', 'Avg Stride Length', 'Max Run Cadence', 'Avg Run Cadence', 'Avg Pace Mins']])
gm.means_
Counter(gm.predict(df[['Best Pace Mins', 'Avg Stride Length', 'Max Run Cadence', 'Avg Run Cadence', 'Avg Pace Mins']]))
```




    Counter({1: 419, 0: 183})




```python
pred = gm.predict(df[['Best Pace Mins', 'Avg Stride Length', 'Max Run Cadence', 'Avg Run Cadence', 'Avg Pace Mins']])
```


```python
Counter(pred)
```




    Counter({1: 419, 0: 183})




```python
df['pred'] = pred
df
```





  <div id="df-c6793656-a031-40c0-aad9-e5d8d4e7053d" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Title</th>
      <th>Distance</th>
      <th>Calories</th>
      <th>Avg HR</th>
      <th>Max HR</th>
      <th>Avg Run Cadence</th>
      <th>Max Run Cadence</th>
      <th>Total Ascent</th>
      <th>Total Descent</th>
      <th>Avg Stride Length</th>
      <th>Min Temp</th>
      <th>Max Temp</th>
      <th>Min Elevation</th>
      <th>Max Elevation</th>
      <th>Total Run Time Mins</th>
      <th>Avg Pace Mins</th>
      <th>Best Pace Mins</th>
      <th>pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>963</th>
      <td>2020-12-31 09:15:19</td>
      <td>Jackson County Running</td>
      <td>10.00</td>
      <td>843.0</td>
      <td>141</td>
      <td>155</td>
      <td>175</td>
      <td>184</td>
      <td>108</td>
      <td>112</td>
      <td>1.31</td>
      <td>69</td>
      <td>77</td>
      <td>-23</td>
      <td>-3</td>
      <td>70.016667</td>
      <td>7.000000</td>
      <td>6.366667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>964</th>
      <td>2020-12-30 09:13:33</td>
      <td>Mobile County Running</td>
      <td>12.00</td>
      <td>1044.0</td>
      <td>147</td>
      <td>165</td>
      <td>174</td>
      <td>184</td>
      <td>200</td>
      <td>194</td>
      <td>1.40</td>
      <td>69</td>
      <td>80</td>
      <td>155</td>
      <td>206</td>
      <td>79.383333</td>
      <td>6.616667</td>
      <td>5.666667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>965</th>
      <td>2020-12-29 09:23:17</td>
      <td>Jackson County Running</td>
      <td>11.16</td>
      <td>865.0</td>
      <td>146</td>
      <td>184</td>
      <td>175</td>
      <td>196</td>
      <td>154</td>
      <td>164</td>
      <td>1.47</td>
      <td>66</td>
      <td>82</td>
      <td>77</td>
      <td>116</td>
      <td>69.566667</td>
      <td>6.233333</td>
      <td>4.450000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>966</th>
      <td>2020-12-28 09:33:42</td>
      <td>Mobile County Running</td>
      <td>9.08</td>
      <td>698.0</td>
      <td>136</td>
      <td>150</td>
      <td>173</td>
      <td>194</td>
      <td>154</td>
      <td>151</td>
      <td>1.33</td>
      <td>68</td>
      <td>82</td>
      <td>11</td>
      <td>76</td>
      <td>63.516667</td>
      <td>7.000000</td>
      <td>6.150000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>967</th>
      <td>2020-12-27 09:20:00</td>
      <td>Mobile County Running</td>
      <td>14.01</td>
      <td>1166.0</td>
      <td>147</td>
      <td>167</td>
      <td>176</td>
      <td>188</td>
      <td>558</td>
      <td>591</td>
      <td>1.41</td>
      <td>62</td>
      <td>77</td>
      <td>98</td>
      <td>284</td>
      <td>91.033333</td>
      <td>6.500000</td>
      <td>5.533333</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>2022-01-05 08:38:41</td>
      <td>Starkville Running</td>
      <td>11.50</td>
      <td>1014.0</td>
      <td>147</td>
      <td>163</td>
      <td>178</td>
      <td>201</td>
      <td>420</td>
      <td>404</td>
      <td>1.30</td>
      <td>57</td>
      <td>75</td>
      <td>223</td>
      <td>402</td>
      <td>79.966667</td>
      <td>6.950000</td>
      <td>6.316667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1971</th>
      <td>2022-01-04 10:30:42</td>
      <td>Noxubee County Running</td>
      <td>11.01</td>
      <td>851.0</td>
      <td>145</td>
      <td>169</td>
      <td>183</td>
      <td>232</td>
      <td>289</td>
      <td>246</td>
      <td>1.42</td>
      <td>51</td>
      <td>69</td>
      <td>130</td>
      <td>201</td>
      <td>67.800000</td>
      <td>6.166667</td>
      <td>4.483333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>2022-01-03 10:29:42</td>
      <td>Starkville Running</td>
      <td>15.01</td>
      <td>1349.0</td>
      <td>148</td>
      <td>171</td>
      <td>179</td>
      <td>190</td>
      <td>807</td>
      <td>774</td>
      <td>1.31</td>
      <td>37</td>
      <td>77</td>
      <td>89</td>
      <td>243</td>
      <td>103.116667</td>
      <td>6.866667</td>
      <td>5.650000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1973</th>
      <td>2022-01-02 09:07:55</td>
      <td>Leon County Running</td>
      <td>8.01</td>
      <td>688.0</td>
      <td>141</td>
      <td>156</td>
      <td>177</td>
      <td>188</td>
      <td>810</td>
      <td>978</td>
      <td>1.29</td>
      <td>77</td>
      <td>86</td>
      <td>-94</td>
      <td>167</td>
      <td>56.416667</td>
      <td>7.050000</td>
      <td>5.716667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1974</th>
      <td>2022-01-01 09:45:26</td>
      <td>Tallahassee Running</td>
      <td>7.00</td>
      <td>593.0</td>
      <td>136</td>
      <td>159</td>
      <td>175</td>
      <td>186</td>
      <td>801</td>
      <td>863</td>
      <td>1.26</td>
      <td>75</td>
      <td>84</td>
      <td>-59</td>
      <td>205</td>
      <td>51.183333</td>
      <td>7.316667</td>
      <td>5.300000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>602 rows × 19 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c6793656-a031-40c0-aad9-e5d8d4e7053d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-c6793656-a031-40c0-aad9-e5d8d4e7053d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c6793656-a031-40c0-aad9-e5d8d4e7053d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-8cb387e3-016b-4a1f-8502-35b003b20e2d">
  <button class="colab-df-quickchart" onclick="quickchart('df-8cb387e3-016b-4a1f-8502-35b003b20e2d')"
            title="Suggest charts."
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
    background-color: #E8F0FE;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: #1967D2;
    height: 32px;
    padding: 0 0 0 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: #E2EBFA;
    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: #174EA6;
  }

  [theme=dark] .colab-df-quickchart {
    background-color: #3B4455;
    fill: #D2E3FC;
  }

  [theme=dark] .colab-df-quickchart:hover {
    background-color: #434B5C;
    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
    fill: #FFFFFF;
  }
</style>

  <script>
    async function quickchart(key) {
      const charts = await google.colab.kernel.invokeFunction(
          'suggestCharts', [key], {});
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-8cb387e3-016b-4a1f-8502-35b003b20e2d button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_28ef4a94-ba14-4808-802e-80d6be6d0927">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_28ef4a94-ba14-4808-802e-80d6be6d0927 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
statistics.median(list(df.groupby([pd.Grouper(key='Date', freq='W')])['pred'].sum()))
```




    6.0




```python
df[df['pred'] == 0].sort_values(by = "Date").tail(50)
```





  <div id="df-22ddeeb3-7c93-4a4c-835d-c9f8e817902a" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Title</th>
      <th>Distance</th>
      <th>Calories</th>
      <th>Avg HR</th>
      <th>Max HR</th>
      <th>Avg Run Cadence</th>
      <th>Max Run Cadence</th>
      <th>Total Ascent</th>
      <th>Total Descent</th>
      <th>Avg Stride Length</th>
      <th>Min Temp</th>
      <th>Max Temp</th>
      <th>Min Elevation</th>
      <th>Max Elevation</th>
      <th>Total Run Time Mins</th>
      <th>Avg Pace Mins</th>
      <th>Best Pace Mins</th>
      <th>pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1923</th>
      <td>2022-02-10 08:47:00</td>
      <td>Boise Running</td>
      <td>4.62</td>
      <td>434.0</td>
      <td>149</td>
      <td>167</td>
      <td>167</td>
      <td>213</td>
      <td>269</td>
      <td>92</td>
      <td>1.31</td>
      <td>53</td>
      <td>68</td>
      <td>2680</td>
      <td>2728</td>
      <td>34.500000</td>
      <td>7.466667</td>
      <td>3.833333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1921</th>
      <td>2022-02-11 17:58:20</td>
      <td>Seattle Running</td>
      <td>0.50</td>
      <td>30.0</td>
      <td>131</td>
      <td>140</td>
      <td>184</td>
      <td>194</td>
      <td>33</td>
      <td>10</td>
      <td>1.53</td>
      <td>60</td>
      <td>66</td>
      <td>17</td>
      <td>52</td>
      <td>2.816667</td>
      <td>5.683333</td>
      <td>4.600000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1920</th>
      <td>2022-02-11 19:08:51</td>
      <td>Seattle Running</td>
      <td>2.12</td>
      <td>176.0</td>
      <td>124</td>
      <td>140</td>
      <td>165</td>
      <td>176</td>
      <td>52</td>
      <td>72</td>
      <td>1.06</td>
      <td>66</td>
      <td>82</td>
      <td>-1</td>
      <td>21</td>
      <td>19.466667</td>
      <td>9.183333</td>
      <td>7.983333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1918</th>
      <td>2022-02-13 09:28:39</td>
      <td>Boise Running</td>
      <td>13.25</td>
      <td>1245.0</td>
      <td>152</td>
      <td>178</td>
      <td>176</td>
      <td>215</td>
      <td>1129</td>
      <td>1227</td>
      <td>1.22</td>
      <td>46</td>
      <td>80</td>
      <td>2409</td>
      <td>3552</td>
      <td>99.966667</td>
      <td>7.550000</td>
      <td>6.283333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1916</th>
      <td>2022-02-15 14:10:45</td>
      <td>Boise Running</td>
      <td>11.36</td>
      <td>866.0</td>
      <td>144</td>
      <td>172</td>
      <td>169</td>
      <td>229</td>
      <td>325</td>
      <td>292</td>
      <td>1.40</td>
      <td>53</td>
      <td>69</td>
      <td>2432</td>
      <td>2518</td>
      <td>76.616667</td>
      <td>6.750000</td>
      <td>2.900000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1912</th>
      <td>2022-02-18 14:09:55</td>
      <td>Boise Running</td>
      <td>10.01</td>
      <td>828.0</td>
      <td>147</td>
      <td>167</td>
      <td>162</td>
      <td>223</td>
      <td>597</td>
      <td>367</td>
      <td>1.28</td>
      <td>59</td>
      <td>73</td>
      <td>2672</td>
      <td>2913</td>
      <td>75.450000</td>
      <td>7.533333</td>
      <td>3.016667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1908</th>
      <td>2022-02-22 14:10:53</td>
      <td>Boise Running</td>
      <td>9.76</td>
      <td>845.0</td>
      <td>151</td>
      <td>174</td>
      <td>173</td>
      <td>223</td>
      <td>295</td>
      <td>308</td>
      <td>1.36</td>
      <td>41</td>
      <td>64</td>
      <td>2665</td>
      <td>2881</td>
      <td>66.033333</td>
      <td>6.766667</td>
      <td>3.750000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1907</th>
      <td>2022-02-23 13:42:56</td>
      <td>Boise Running</td>
      <td>8.00</td>
      <td>669.0</td>
      <td>142</td>
      <td>164</td>
      <td>175</td>
      <td>208</td>
      <td>164</td>
      <td>190</td>
      <td>1.31</td>
      <td>42</td>
      <td>73</td>
      <td>2715</td>
      <td>2781</td>
      <td>56.083333</td>
      <td>7.000000</td>
      <td>6.100000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1906</th>
      <td>2022-02-24 14:52:31</td>
      <td>Albuquerque Running</td>
      <td>5.00</td>
      <td>454.0</td>
      <td>142</td>
      <td>161</td>
      <td>161</td>
      <td>215</td>
      <td>135</td>
      <td>115</td>
      <td>1.28</td>
      <td>55</td>
      <td>80</td>
      <td>4936</td>
      <td>4977</td>
      <td>38.916667</td>
      <td>7.783333</td>
      <td>3.700000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>2022-02-25 16:46:29</td>
      <td>Albuquerque Running</td>
      <td>0.49</td>
      <td>31.0</td>
      <td>134</td>
      <td>144</td>
      <td>183</td>
      <td>205</td>
      <td>23</td>
      <td>23</td>
      <td>1.62</td>
      <td>64</td>
      <td>66</td>
      <td>4953</td>
      <td>4969</td>
      <td>2.666667</td>
      <td>5.433333</td>
      <td>4.100000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1903</th>
      <td>2022-02-25 17:39:15</td>
      <td>Albuquerque Running</td>
      <td>1.92</td>
      <td>179.0</td>
      <td>131</td>
      <td>140</td>
      <td>167</td>
      <td>178</td>
      <td>194</td>
      <td>13</td>
      <td>1.10</td>
      <td>64</td>
      <td>82</td>
      <td>5071</td>
      <td>5085</td>
      <td>16.900000</td>
      <td>8.800000</td>
      <td>2.600000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1902</th>
      <td>2022-02-26 12:49:29</td>
      <td>Albuquerque Running</td>
      <td>2.12</td>
      <td>189.0</td>
      <td>132</td>
      <td>142</td>
      <td>175</td>
      <td>186</td>
      <td>95</td>
      <td>98</td>
      <td>1.13</td>
      <td>60</td>
      <td>75</td>
      <td>5151</td>
      <td>5194</td>
      <td>17.250000</td>
      <td>8.133333</td>
      <td>6.550000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1901</th>
      <td>2022-02-26 13:15:52</td>
      <td>Running</td>
      <td>0.44</td>
      <td>24.0</td>
      <td>127</td>
      <td>134</td>
      <td>185</td>
      <td>196</td>
      <td>36</td>
      <td>13</td>
      <td>1.62</td>
      <td>64</td>
      <td>66</td>
      <td>5161</td>
      <td>5191</td>
      <td>2.366667</td>
      <td>5.350000</td>
      <td>4.666667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1900</th>
      <td>2022-02-26 14:16:29</td>
      <td>Albuquerque Running</td>
      <td>2.00</td>
      <td>200.0</td>
      <td>139</td>
      <td>156</td>
      <td>173</td>
      <td>184</td>
      <td>180</td>
      <td>236</td>
      <td>1.11</td>
      <td>68</td>
      <td>78</td>
      <td>5043</td>
      <td>5159</td>
      <td>16.766667</td>
      <td>8.400000</td>
      <td>7.150000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1895</th>
      <td>2022-03-04 14:12:51</td>
      <td>Boise Running</td>
      <td>13.22</td>
      <td>1196.0</td>
      <td>156</td>
      <td>176</td>
      <td>178</td>
      <td>208</td>
      <td>1096</td>
      <td>1073</td>
      <td>1.33</td>
      <td>53</td>
      <td>68</td>
      <td>2610</td>
      <td>2957</td>
      <td>89.566667</td>
      <td>6.783333</td>
      <td>3.633333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1891</th>
      <td>2022-03-08 14:10:13</td>
      <td>Boise Running</td>
      <td>13.99</td>
      <td>1078.0</td>
      <td>153</td>
      <td>175</td>
      <td>182</td>
      <td>195</td>
      <td>361</td>
      <td>243</td>
      <td>1.51</td>
      <td>50</td>
      <td>69</td>
      <td>2678</td>
      <td>2778</td>
      <td>82.016667</td>
      <td>5.866667</td>
      <td>4.783333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1887</th>
      <td>2022-03-11 14:14:13</td>
      <td>Boise Running</td>
      <td>14.00</td>
      <td>1244.0</td>
      <td>153</td>
      <td>174</td>
      <td>180</td>
      <td>212</td>
      <td>1083</td>
      <td>932</td>
      <td>1.32</td>
      <td>53</td>
      <td>73</td>
      <td>2713</td>
      <td>3060</td>
      <td>94.650000</td>
      <td>6.766667</td>
      <td>3.716667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1886</th>
      <td>2022-03-12 09:07:49</td>
      <td>Boise Running</td>
      <td>15.00</td>
      <td>1258.0</td>
      <td>147</td>
      <td>172</td>
      <td>178</td>
      <td>215</td>
      <td>1749</td>
      <td>1690</td>
      <td>1.25</td>
      <td>60</td>
      <td>77</td>
      <td>2602</td>
      <td>3829</td>
      <td>108.133333</td>
      <td>7.200000</td>
      <td>4.916667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1882</th>
      <td>2022-03-15 09:40:36</td>
      <td>Boise Running</td>
      <td>11.00</td>
      <td>831.0</td>
      <td>141</td>
      <td>175</td>
      <td>161</td>
      <td>215</td>
      <td>128</td>
      <td>184</td>
      <td>1.52</td>
      <td>50</td>
      <td>69</td>
      <td>2603</td>
      <td>2734</td>
      <td>72.800000</td>
      <td>6.616667</td>
      <td>4.116667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1878</th>
      <td>2022-03-18 09:42:02</td>
      <td>Buncombe County Running</td>
      <td>12.47</td>
      <td>1188.0</td>
      <td>152</td>
      <td>176</td>
      <td>175</td>
      <td>210</td>
      <td>1168</td>
      <td>1191</td>
      <td>1.22</td>
      <td>57</td>
      <td>73</td>
      <td>1892</td>
      <td>2131</td>
      <td>93.866667</td>
      <td>7.533333</td>
      <td>4.366667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1874</th>
      <td>2022-03-23 11:12:18</td>
      <td>Boise Running</td>
      <td>10.32</td>
      <td>725.0</td>
      <td>139</td>
      <td>159</td>
      <td>177</td>
      <td>226</td>
      <td>554</td>
      <td>292</td>
      <td>1.34</td>
      <td>66</td>
      <td>80</td>
      <td>2663</td>
      <td>2856</td>
      <td>69.400000</td>
      <td>6.716667</td>
      <td>3.533333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>2022-03-29 14:11:12</td>
      <td>Boise Running</td>
      <td>11.00</td>
      <td>801.0</td>
      <td>139</td>
      <td>168</td>
      <td>153</td>
      <td>232</td>
      <td>673</td>
      <td>627</td>
      <td>1.41</td>
      <td>69</td>
      <td>80</td>
      <td>2681</td>
      <td>2865</td>
      <td>79.350000</td>
      <td>7.216667</td>
      <td>3.600000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1865</th>
      <td>2022-03-31 13:22:23</td>
      <td>Los Altos Hills Running</td>
      <td>5.06</td>
      <td>461.0</td>
      <td>142</td>
      <td>161</td>
      <td>172</td>
      <td>208</td>
      <td>285</td>
      <td>230</td>
      <td>1.26</td>
      <td>69</td>
      <td>86</td>
      <td>266</td>
      <td>398</td>
      <td>37.450000</td>
      <td>7.400000</td>
      <td>4.200000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1864</th>
      <td>2022-04-01 18:39:22</td>
      <td>Santa Clara County Running</td>
      <td>2.38</td>
      <td>291.0</td>
      <td>147</td>
      <td>164</td>
      <td>166</td>
      <td>181</td>
      <td>30</td>
      <td>33</td>
      <td>1.08</td>
      <td>68</td>
      <td>80</td>
      <td>74</td>
      <td>96</td>
      <td>21.516667</td>
      <td>9.033333</td>
      <td>7.016667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1863</th>
      <td>2022-04-01 19:13:15</td>
      <td>Santa Clara County Running</td>
      <td>0.45</td>
      <td>29.0</td>
      <td>136</td>
      <td>145</td>
      <td>181</td>
      <td>194</td>
      <td>10</td>
      <td>36</td>
      <td>1.57</td>
      <td>68</td>
      <td>69</td>
      <td>30</td>
      <td>42</td>
      <td>2.550000</td>
      <td>5.616667</td>
      <td>4.783333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1862</th>
      <td>2022-04-01 20:07:17</td>
      <td>Santa Clara County Running</td>
      <td>3.12</td>
      <td>421.0</td>
      <td>159</td>
      <td>174</td>
      <td>167</td>
      <td>178</td>
      <td>43</td>
      <td>92</td>
      <td>1.07</td>
      <td>66</td>
      <td>75</td>
      <td>-138</td>
      <td>-84</td>
      <td>28.050000</td>
      <td>9.000000</td>
      <td>6.566667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1861</th>
      <td>2022-04-02 08:15:29</td>
      <td>San Mateo County Running</td>
      <td>12.00</td>
      <td>1248.0</td>
      <td>149</td>
      <td>168</td>
      <td>174</td>
      <td>195</td>
      <td>1759</td>
      <td>1837</td>
      <td>1.11</td>
      <td>55</td>
      <td>73</td>
      <td>1764</td>
      <td>2210</td>
      <td>100.033333</td>
      <td>8.333333</td>
      <td>5.483333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1857</th>
      <td>2022-04-05 14:11:33</td>
      <td>Boise Running</td>
      <td>12.36</td>
      <td>1023.0</td>
      <td>152</td>
      <td>171</td>
      <td>176</td>
      <td>232</td>
      <td>492</td>
      <td>358</td>
      <td>1.40</td>
      <td>53</td>
      <td>69</td>
      <td>2436</td>
      <td>2591</td>
      <td>79.533333</td>
      <td>6.433333</td>
      <td>3.533333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1851</th>
      <td>2022-04-11 15:11:23</td>
      <td>Boise Running</td>
      <td>10.17</td>
      <td>904.0</td>
      <td>148</td>
      <td>178</td>
      <td>167</td>
      <td>208</td>
      <td>604</td>
      <td>607</td>
      <td>1.34</td>
      <td>55</td>
      <td>68</td>
      <td>2704</td>
      <td>2900</td>
      <td>71.450000</td>
      <td>7.033333</td>
      <td>3.816667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1849</th>
      <td>2022-04-13 08:31:03</td>
      <td>Boise Running</td>
      <td>5.00</td>
      <td>422.0</td>
      <td>132</td>
      <td>164</td>
      <td>155</td>
      <td>201</td>
      <td>59</td>
      <td>92</td>
      <td>1.25</td>
      <td>53</td>
      <td>77</td>
      <td>2679</td>
      <td>2734</td>
      <td>41.200000</td>
      <td>8.233333</td>
      <td>4.383333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1846</th>
      <td>2022-04-14 22:22:25</td>
      <td>Walnut Running</td>
      <td>1.26</td>
      <td>100.0</td>
      <td>116</td>
      <td>125</td>
      <td>155</td>
      <td>174</td>
      <td>26</td>
      <td>10</td>
      <td>1.10</td>
      <td>66</td>
      <td>78</td>
      <td>713</td>
      <td>728</td>
      <td>11.966667</td>
      <td>9.466667</td>
      <td>8.100000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1845</th>
      <td>2022-04-15 09:05:24</td>
      <td>West Covina Running</td>
      <td>5.78</td>
      <td>364.0</td>
      <td>118</td>
      <td>132</td>
      <td>170</td>
      <td>184</td>
      <td>243</td>
      <td>315</td>
      <td>1.14</td>
      <td>68</td>
      <td>82</td>
      <td>499</td>
      <td>642</td>
      <td>47.850000</td>
      <td>8.283333</td>
      <td>6.250000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1844</th>
      <td>2022-04-16 07:24:25</td>
      <td>La Verne Running</td>
      <td>11.17</td>
      <td>919.0</td>
      <td>135</td>
      <td>161</td>
      <td>170</td>
      <td>223</td>
      <td>1388</td>
      <td>1526</td>
      <td>1.16</td>
      <td>55</td>
      <td>78</td>
      <td>1072</td>
      <td>1852</td>
      <td>91.250000</td>
      <td>8.166667</td>
      <td>4.550000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1841</th>
      <td>2022-04-19 14:11:28</td>
      <td>Boise Running</td>
      <td>9.00</td>
      <td>790.0</td>
      <td>144</td>
      <td>172</td>
      <td>146</td>
      <td>212</td>
      <td>423</td>
      <td>295</td>
      <td>1.33</td>
      <td>62</td>
      <td>77</td>
      <td>2653</td>
      <td>2864</td>
      <td>71.366667</td>
      <td>7.933333</td>
      <td>3.516667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1837</th>
      <td>2022-04-22 14:10:05</td>
      <td>Boise Running</td>
      <td>11.00</td>
      <td>779.0</td>
      <td>140</td>
      <td>169</td>
      <td>173</td>
      <td>234</td>
      <td>525</td>
      <td>492</td>
      <td>1.38</td>
      <td>66</td>
      <td>78</td>
      <td>2625</td>
      <td>2782</td>
      <td>73.116667</td>
      <td>6.650000</td>
      <td>3.100000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1836</th>
      <td>2022-04-23 09:06:59</td>
      <td>Boise Running</td>
      <td>13.04</td>
      <td>1033.0</td>
      <td>139</td>
      <td>159</td>
      <td>176</td>
      <td>215</td>
      <td>1732</td>
      <td>1742</td>
      <td>1.21</td>
      <td>50</td>
      <td>78</td>
      <td>2705</td>
      <td>3895</td>
      <td>98.400000</td>
      <td>7.550000</td>
      <td>5.650000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1834</th>
      <td>2022-04-25 15:08:30</td>
      <td>Boise Running</td>
      <td>6.97</td>
      <td>689.0</td>
      <td>151</td>
      <td>167</td>
      <td>172</td>
      <td>220</td>
      <td>351</td>
      <td>351</td>
      <td>1.24</td>
      <td>73</td>
      <td>84</td>
      <td>2673</td>
      <td>2856</td>
      <td>52.583333</td>
      <td>7.550000</td>
      <td>6.800000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1833</th>
      <td>2022-04-26 14:10:18</td>
      <td>Boise Running</td>
      <td>11.48</td>
      <td>787.0</td>
      <td>141</td>
      <td>167</td>
      <td>170</td>
      <td>226</td>
      <td>466</td>
      <td>469</td>
      <td>1.35</td>
      <td>66</td>
      <td>84</td>
      <td>2663</td>
      <td>2851</td>
      <td>79.533333</td>
      <td>6.933333</td>
      <td>3.516667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1830</th>
      <td>2022-04-29 08:26:40</td>
      <td>Boise Running</td>
      <td>5.01</td>
      <td>409.0</td>
      <td>138</td>
      <td>161</td>
      <td>170</td>
      <td>206</td>
      <td>98</td>
      <td>82</td>
      <td>1.27</td>
      <td>57</td>
      <td>73</td>
      <td>2705</td>
      <td>2743</td>
      <td>37.183333</td>
      <td>7.433333</td>
      <td>4.216667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1829</th>
      <td>2022-04-30 21:16:08</td>
      <td>Clovis Running</td>
      <td>2.33</td>
      <td>195.0</td>
      <td>135</td>
      <td>150</td>
      <td>172</td>
      <td>231</td>
      <td>36</td>
      <td>43</td>
      <td>1.26</td>
      <td>69</td>
      <td>75</td>
      <td>318</td>
      <td>332</td>
      <td>17.300000</td>
      <td>7.433333</td>
      <td>6.766667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1828</th>
      <td>2022-04-30 21:45:40</td>
      <td>Clovis Running</td>
      <td>0.44</td>
      <td>24.0</td>
      <td>130</td>
      <td>136</td>
      <td>183</td>
      <td>201</td>
      <td>43</td>
      <td>3</td>
      <td>1.65</td>
      <td>69</td>
      <td>69</td>
      <td>333</td>
      <td>346</td>
      <td>2.316667</td>
      <td>5.333333</td>
      <td>4.566667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1827</th>
      <td>2022-04-30 22:29:46</td>
      <td>Clovis Running</td>
      <td>2.38</td>
      <td>173.0</td>
      <td>122</td>
      <td>137</td>
      <td>164</td>
      <td>184</td>
      <td>180</td>
      <td>115</td>
      <td>1.18</td>
      <td>71</td>
      <td>78</td>
      <td>325</td>
      <td>401</td>
      <td>19.783333</td>
      <td>8.300000</td>
      <td>6.483333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1826</th>
      <td>2022-05-02 15:50:12</td>
      <td>Boise Running</td>
      <td>7.00</td>
      <td>657.0</td>
      <td>152</td>
      <td>170</td>
      <td>173</td>
      <td>229</td>
      <td>197</td>
      <td>233</td>
      <td>1.31</td>
      <td>48</td>
      <td>71</td>
      <td>2637</td>
      <td>2723</td>
      <td>49.700000</td>
      <td>7.100000</td>
      <td>3.650000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1825</th>
      <td>2022-05-03 14:13:39</td>
      <td>Boise Running</td>
      <td>11.64</td>
      <td>939.0</td>
      <td>148</td>
      <td>172</td>
      <td>178</td>
      <td>218</td>
      <td>610</td>
      <td>423</td>
      <td>1.35</td>
      <td>64</td>
      <td>77</td>
      <td>2713</td>
      <td>2983</td>
      <td>77.166667</td>
      <td>6.633333</td>
      <td>3.433333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1822</th>
      <td>2022-05-06 10:09:25</td>
      <td>Boise Running</td>
      <td>10.16</td>
      <td>703.0</td>
      <td>136</td>
      <td>163</td>
      <td>163</td>
      <td>236</td>
      <td>482</td>
      <td>430</td>
      <td>1.30</td>
      <td>68</td>
      <td>80</td>
      <td>2672</td>
      <td>2851</td>
      <td>76.200000</td>
      <td>7.500000</td>
      <td>3.300000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1819</th>
      <td>2022-05-09 15:10:00</td>
      <td>Boise Running</td>
      <td>9.03</td>
      <td>651.0</td>
      <td>134</td>
      <td>166</td>
      <td>173</td>
      <td>219</td>
      <td>194</td>
      <td>200</td>
      <td>1.34</td>
      <td>53</td>
      <td>73</td>
      <td>2654</td>
      <td>2740</td>
      <td>62.150000</td>
      <td>6.883333</td>
      <td>3.550000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1817</th>
      <td>2022-05-11 08:27:01</td>
      <td>Boise Running</td>
      <td>4.90</td>
      <td>415.0</td>
      <td>138</td>
      <td>162</td>
      <td>175</td>
      <td>208</td>
      <td>233</td>
      <td>118</td>
      <td>1.25</td>
      <td>53</td>
      <td>75</td>
      <td>2697</td>
      <td>2753</td>
      <td>35.883333</td>
      <td>7.316667</td>
      <td>4.233333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1814</th>
      <td>2022-05-12 23:02:28</td>
      <td>Clovis Running</td>
      <td>2.64</td>
      <td>205.0</td>
      <td>122</td>
      <td>137</td>
      <td>169</td>
      <td>182</td>
      <td>30</td>
      <td>39</td>
      <td>1.09</td>
      <td>71</td>
      <td>78</td>
      <td>305</td>
      <td>333</td>
      <td>22.966667</td>
      <td>8.683333</td>
      <td>7.266667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1811</th>
      <td>2022-05-14 20:04:07</td>
      <td>Clovis Running</td>
      <td>0.40</td>
      <td>19.0</td>
      <td>121</td>
      <td>131</td>
      <td>185</td>
      <td>192</td>
      <td>7</td>
      <td>7</td>
      <td>1.66</td>
      <td>78</td>
      <td>78</td>
      <td>327</td>
      <td>336</td>
      <td>2.100000</td>
      <td>5.233333</td>
      <td>4.566667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1810</th>
      <td>2022-05-14 20:45:49</td>
      <td>Clovis Running</td>
      <td>1.13</td>
      <td>108.0</td>
      <td>128</td>
      <td>144</td>
      <td>165</td>
      <td>178</td>
      <td>30</td>
      <td>26</td>
      <td>1.04</td>
      <td>86</td>
      <td>87</td>
      <td>321</td>
      <td>351</td>
      <td>10.666667</td>
      <td>9.433333</td>
      <td>7.750000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-22ddeeb3-7c93-4a4c-835d-c9f8e817902a')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-22ddeeb3-7c93-4a4c-835d-c9f8e817902a button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-22ddeeb3-7c93-4a4c-835d-c9f8e817902a');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-31a2933c-27ab-4914-98d5-dcfc4d03af58">
  <button class="colab-df-quickchart" onclick="quickchart('df-31a2933c-27ab-4914-98d5-dcfc4d03af58')"
            title="Suggest charts."
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
    background-color: #E8F0FE;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: #1967D2;
    height: 32px;
    padding: 0 0 0 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: #E2EBFA;
    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: #174EA6;
  }

  [theme=dark] .colab-df-quickchart {
    background-color: #3B4455;
    fill: #D2E3FC;
  }

  [theme=dark] .colab-df-quickchart:hover {
    background-color: #434B5C;
    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
    fill: #FFFFFF;
  }
</style>

  <script>
    async function quickchart(key) {
      const charts = await google.colab.kernel.invokeFunction(
          'suggestCharts', [key], {});
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-31a2933c-27ab-4914-98d5-dcfc4d03af58 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python

```
