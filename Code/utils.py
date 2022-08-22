import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from math import radians
from math import tan,atan,acos,sin,cos,asin,sqrt
from scipy.spatial.distance import pdist, squareform
import time
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import numpy as np

dst_folder = "C:\\Users\\Reset me please\\Desktop\\PMI\\Processed_Data1\\"
def join_data(dst_path):
    arr = os.listdir(dst_folder)
    df_final = pd.DataFrame()
    for each in arr:
        df1= pd.read_csv(os.path.join(dst_folder,each))
        df1["Cab_Name"]= each
        df_final = pd.concat([df_final, df1])
    return(df_final)

def minutes_convert(x):
    if x in range(0,20):
        return(1)
    elif x in range(20,40):
        return(2)
    elif x in range(40,61):
        return(3)
    

def haversine(lonlat1, lonlat2):#
    lat1, lon1 = lonlat1
    lat2, lon2 = lonlat2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000

def train_test(df_final, split):
    df_final["Date"] = pd.to_datetime(df_final["Date"])
    df_final["Date_1"] = pd.to_datetime(df_final["Date_1"])
    df_final["Minutes"]= df_final["Date"].dt.strftime('%M')
    df_final["Minutes_1"]= df_final["Date_1"].dt.strftime('%M')
    df_final["Minutes_Identifier"] = df_final["Minutes"].astype("int").apply( lambda x : minutes_convert(x))
    df_final["Minutes_Identifier_1"] = df_final["Minutes_1"].astype("int").apply( lambda x : minutes_convert(x))
    df_final = df_final[["Latitude","Longitude","Hour","Day","Minutes_Identifier","Latitude_1","Longitude_1"]]
    
    X =df_final[["Latitude","Longitude","Hour","Day","Minutes_Identifier"]]
    X =X.reset_index().drop("index", axis=1)
    y = df_final[["Latitude_1","Longitude_1"]]
    y = y.reset_index().drop("index", axis=1)
    
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split( X, y, test_size=split, random_state=0)
    
    X_train, X_test, y_train, y_test, scaler =scale_data(X_train_o.copy(), X_test_o.copy(), y_train_o.copy(), y_test_o.copy())
    return(X_train_o, X_test_o, y_train_o, y_test_o,X_train, X_test, y_train, y_test, scaler)

def scale_data(X_train, X_test, y_train, y_test):
    X_train["Latitude"] = X_train["Latitude"].astype("float32")
    X_train["Longitude"] = X_train["Longitude"].astype("float32")
    X_train["Minutes_Identifier"]=X_train["Minutes_Identifier"].astype("int")
    
    X_train["Latitude"] = X_train["Latitude"]/90
    X_train["Longitude"] = X_train["Longitude"]/180
    scaler = StandardScaler()
    X_train[['Hour', 'Day', 'Minutes_Identifier']] = scaler.fit_transform(X_train[['Hour', 'Day', 'Minutes_Identifier']])
    X_train["Latitude"] = X_train["Latitude"].astype("float32")
    X_train["Longitude"] = X_train["Longitude"].astype("float32")
    X_train["Hour"] = X_train["Hour"].astype("float32")
    X_train["Day"] = X_train["Day"].astype("float32")
    X_train["Minutes_Identifier"] = X_train["Minutes_Identifier"].astype("float32")
    
    #.................................................................................
    X_test["Latitude"] = X_test["Latitude"].astype("float32")
    X_test["Longitude"] = X_test["Longitude"].astype("float32")
    X_test["Minutes_Identifier"]=X_test["Minutes_Identifier"].astype("int")
    
    X_test["Latitude"] = X_test["Latitude"]/90
    X_test["Longitude"] = X_test["Longitude"]/180
    
    X_test[['Hour', 'Day', 'Minutes_Identifier']] = scaler.transform(X_test[['Hour', 'Day', 'Minutes_Identifier']])
    X_test["Latitude"] = X_test["Latitude"].astype("float32")
    X_test["Longitude"] = X_test["Longitude"].astype("float32")
    X_test["Hour"] = X_test["Hour"].astype("float32")
    X_test["Day"] = X_test["Day"].astype("float32")
    X_test["Minutes_Identifier"] = X_test["Minutes_Identifier"].astype("float32")
    
    
    #-...................................................
    
    y_train["Latitude_1"] = y_train["Latitude_1"].astype("float32")
    y_train["Longitude_1"] = y_train["Longitude_1"].astype("float32")
#     y["Latitude_1"] = y["Latitude_1"]*100000
#     y["Longitude_1"] = y["Longitude_1"]*100000
    y_train = y_train.reindex(columns=["Latitude_1","Longitude_1"])
    
    
    y_test["Latitude_1"] = y_test["Latitude_1"].astype("float32")
    y_test["Longitude_1"] = y_test["Longitude_1"].astype("float32")
#     y["Latitude_1"] = y["Latitude_1"]*100000
#     y["Longitude_1"] = y["Longitude_1"]*100000
    y_test = y_test.reindex(columns=["Latitude_1","Longitude_1"])
    
    
    X_train = X_train.reset_index().drop("index", axis=1)
    X_test = X_test.reset_index().drop("index", axis=1)
    y_train = y_train.reset_index().drop("index", axis=1)
    y_test = y_test.reset_index().drop("index", axis=1)
    return(X_train, X_test, y_train, y_test, scaler)


# def train_test(df_final, split):
#     df_final["Date"] = pd.to_datetime(df_final["Date"])
#     df_final["Date_1"] = pd.to_datetime(df_final["Date_1"])
#     df_final["Minutes"]= df_final["Date"].dt.strftime('%M')
#     df_final["Minutes_1"]= df_final["Date_1"].dt.strftime('%M')
#     df_final["Minutes_Identifier"] = df_final["Minutes"].astype("int").apply( lambda x : minutes_convert(x))
#     df_final["Minutes_Identifier_1"] = df_final["Minutes_1"].astype("int").apply( lambda x : minutes_convert(x))
#     df_final = df_final[["Latitude","Longitude","Hour","Day","Minutes","Latitude_1","Longitude_1"]]
    
#     X =df_final[["Latitude","Longitude","Hour","Day","Minutes"]]
#     X =X.reset_index().drop("index", axis=1)
#     y = df_final[["Latitude_1","Longitude_1"]]
#     y = y.reset_index().drop("index", axis=1)
#     X["Latitude"] = X["Latitude"].astype("float32")
#     X["Longitude"] = X["Longitude"].astype("float32")
#     X["Minutes"]=X["Minutes"].astype("int")
    
#     X["Latitude"] = X["Latitude"]/90
#     X["Longitude"] = X["Longitude"]/180
#     scaler = StandardScaler()
#     X[['Hour', 'Day', 'Minutes']] = scaler.fit_transform(X[['Hour', 'Day', 'Minutes']])
#     X["Latitude"] = X["Latitude"].astype("float32")
#     X["Longitude"] = X["Longitude"].astype("float32")
#     X["Hour"] = X["Hour"].astype("float32")
#     X["Day"] = X["Day"].astype("float32")
#     X["Minutes"] = X["Minutes"].astype("float32")
#     y["Latitude_1"] = y["Latitude_1"].astype("float32")
#     y["Longitude_1"] = y["Longitude_1"].astype("float32")
# #     y["Latitude_1"] = y["Latitude_1"]*100000
# #     y["Longitude_1"] = y["Longitude_1"]*100000
#     y = y.reindex(columns=["Latitude_1","Longitude_1"])
    
#     X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=split, random_state=0)
#     X_train = X_train.reset_index().drop("index", axis=1)
#     X_test = X_test.reset_index().drop("index", axis=1)
#     y_train = y_train.reset_index().drop("index", axis=1)
#     y_test = y_test.reset_index().drop("index", axis=1)
#     return(X_train, X_test, y_train, y_test, scaler)

def hervy_dist(y_pre, y_true):
    y_pre = pd.DataFrame(y_pre, columns =["Latitude_pre","Longitude_pre"])
    df = pd.concat([y_pre, y_true], axis=1)
#     df =df/100000
    df["Distance"]=df.apply(lambda x :haversine([x["Latitude_pre"],x["Longitude_pre"]], [x["Latitude_1"],x["Longitude_1"]] ), axis=1)
    return(df)

def outlier_removal(df1):
    df1.columns = ["Latitude", "Longitude", "Occupancy", "Time_Stamp"]
    df1["Date"] = df1["Time_Stamp"].apply(lambda x: time.strftime("'%Y-%m-%d %H:%M:%S'", time.localtime(x)))
    df1["Date"] = pd.to_datetime(df1["Date"])
    df1 = df1.sort_values(by='Date')
    df1_1= df1[["Latitude", "Longitude","Date"]][1:].copy()
    df1_1 = df1_1.reset_index().drop("index", axis=1)
    df1 = df1[:-1]
    df1 = df1.reset_index().drop("index", axis=1)
    df1[["Latitude_1", "Longitude_1","Date_1"]] = df1_1
    df1["Distance"] = df1.apply(lambda x :haversine([x.Latitude,x.Longitude], [x.Latitude_1,x.Longitude_1])/1000,  axis=1)
    df1["Time_Spent"]= df1.apply(lambda x : ((x.Date_1-x.Date).total_seconds())/3600,axis=1)
    df1["Speed"] = df1.apply(lambda x : x.Distance/ x.Time_Spent, axis=1)
    return(df1[df1["Speed"]<=50])

def train_data(df):
    df_transition = df[df["Occupancy"].diff()!=0]
    df_transition = df_transition[["Latitude","Longitude","Occupancy","Date"]]
    df_transition["Hour"]= df_transition["Date"].dt.strftime('%H')
    df_transition["Day"]=df_transition["Date"].apply(lambda x : x.weekday())
    df_transition_1= df_transition[["Latitude","Longitude","Occupancy","Date","Hour","Day"]][1:].copy()
    df_transition_1= df_transition_1.reset_index().drop("index", axis=1)
    df_transition = df_transition[:-1]
    df_transition = df_transition.reset_index().drop("index", axis=1)
    df_transition[["Latitude_1","Longitude_1","Occupancy_1","Date_1","Hour_1","Day_1"]] = df_transition_1
    return(df_transition[df_transition["Occupancy"]==0])


def taxi_demand_cluster(df_final, hour, minutes,day, eps, samples ):
    df_final["Date"] = pd.to_datetime(df_final["Date"])
    df_final["Date_1"] = pd.to_datetime(df_final["Date_1"])
    df_final["Minutes"]= df_final["Date"].dt.strftime('%M')
    df_final["Minutes_1"]= df_final["Date_1"].dt.strftime('%M')
    
    df_final["Minutes_Identifier"] = df_final["Minutes"].astype("int").apply( lambda x : minutes_convert(x))
    df_final["Minutes_Identifier_1"] = df_final["Minutes_1"].astype("int").apply( lambda x : minutes_convert(x))
    if minutes !=0:
        df_clustering = df_final[(df_final["Hour_1"]==hour) &(df_final["Minutes_Identifier_1"]==minutes)&(df_final["Day_1"]==day)].copy()
    else:
        df_clustering = df_final[(df_final["Hour_1"]==hour) &(df_final["Day_1"]==day)].copy()
        
        
    df_clustering.reset_index(inplace=True)
    df_clustering = df_clustering[["Latitude_1", "Longitude_1"]].dropna(axis=0,how='all')
    distance_matrix = squareform(pdist(df_clustering, (lambda u, v: haversine(u, v))))
    db = DBSCAN(eps=eps, min_samples=samples, metric='precomputed').fit_predict(distance_matrix)
    df_clustering['label'] = db
    df_clustering = df_clustering[df_clustering["label"]!=-1]
    # df_test = pd.DataFrame(distance_matrix).filter(items = list(df_clustering.index) , axis=0)
    # df_test =df_test.filter(items = list(df_clustering.index) , axis=1)
    
    cluster_centroids=[]
    num_clusters = len(set(df_clustering['label']))
    #print('Number of clusters: {:,}'.format(num_clusters))
    clusters = pd.Series([df_clustering[df_clustering['label']==n] for n in range(num_clusters)])
    for each in range(num_clusters):
        lat =MultiPoint(clusters[each][["Latitude_1","Longitude_1"]].values).centroid.xy[0][0]
        long = MultiPoint(clusters[each][["Latitude_1","Longitude_1"]].values).centroid.xy[1][0]
        #print("Centroid for Label: ",each,lat, long)
        cluster_centroids.append((each,lat, long))
    
    df_centroids = pd.DataFrame(cluster_centroids)
    df_centroids.columns = ["label", "Latitude_1","Longitude_1"]
    return(df_clustering,df_centroids)

def plt_map(df,f_name):
    from pyproj import CRS

    crs=CRS('EPSG:4326').to_proj4()
    street_map  = gpd.read_file("sf2/San_Francisco_Bay_Region_Roadways.shp")
    geometry = [Point(xy) for xy in zip(  df["Longitude_1"],df["Latitude_1"])]
    geo_df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    sf_001 = street_map[  (street_map["countyfp"]=="075") | (street_map["countyfp"]=="081")| (street_map["countyfp"]=="001")]
    fig,ax = plt.subplots(figsize =(40,40))
    sf_001.plot(ax=ax, alpha =0.2, color ="grey")
    geo_df.plot(ax=ax,marker='o',markersize=100,aspect=1, column="label",legend=True,categorical=True,cmap='tab20')
    plt.tight_layout()
    #plt.ioff()
    #plt.axis('off')
    
    plt.savefig(f_name)
    #plt.close(fig)
    return(print("Plot"))

def predict_lat_long(test_lat_long, df_centroids):
    distance=[]
    for lat, long in zip(df_centroids["Latitude_1"], df_centroids["Longitude_1"]):
        distance.append(haversine(test_lat_long, [lat,long]))
    df_centroids["Distance"]=distance
    df_centroids =df_centroids[["Latitude_1","Longitude_1","Distance"]][:2]
    
    test_lat_long.append(0)
    df = df_centroids[["Latitude_1","Longitude_1"]].copy()
    if df.shape[0]==0:
        return(pd.DataFrame(),pd.DataFrame())
    else:
        labels=[1,2]
        df["label"]=labels
        df = pd.concat([df,pd.DataFrame(np.array(test_lat_long).reshape(1,3), columns=["Latitude_1","Longitude_1","label"])], axis=0)
    return(df_centroids.sort_values(by ="Distance"), df)
