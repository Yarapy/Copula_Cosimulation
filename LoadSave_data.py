##Modules##
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfile


def Load_columns(column_number,selected_columns,skipped_data=0,delimiter="\t",windowname='Select File'):
    #Show Tkinter dialog box (ask which file you want to open)
    root = tk.Tk()
    root.withdraw()
    filename_csv = askopenfilename(title=windowname,filetypes=[("Dat files","*.dat")])
    column_names = [i for i in range(1,column_number+1,1)]
    #Skiprows is the line from which the data begins.
    Data= pd.read_csv(filename_csv, sep=delimiter, skiprows=column_number+2+skipped_data,names=column_names )
    return np.array(Data[selected_columns])

def Save_columns(variable,columns=["coord1", "coord2", "var"]):
    #Show Tkinter dialog box (ask which file you want to open)
    root = tk.Tk()
    root.withdraw()
    filename_csv = asksaveasfile(filetypes=[("Dat files","*.dat")])
    Save_Data(filename_csv,variable,columns=columns)

def Save_Data(filename,variable,columns=["coord1", "coord2", "var"],delimiter=" "):
    column_number=len(variable[1])
    column_names=''
    for i in range(0,column_number,1):
        if column_number==len(columns):
            column_names = column_names+'\n'+columns[i]
        else:
            column_names = column_names+'\n'+str(i)
    np.savetxt(filename,variable,header=filename+'\n'+str(column_number)+column_names,delimiter=delimiter, comments='')

def Save_Vectors(filename,variable,vector_lenght,columns=["coord1", "coord2", "var"],delimiter=" "):
    column_number=len(variable[1])
    column_names=''
    for i in range(0,column_number,1):
        if column_number==len(columns):
            column_names = column_names+'\n'+columns[i]
        else:
            column_names = column_names+'\n'+str(i)
    np.savetxt(filename,variable,header=filename+'\n'+str(column_number)+column_names+'\n'+str(vector_lenght),delimiter=delimiter, comments='')

def Load_Vectors(column_number,selected_columns,skipped_data=0,delimiter="\t",windowname='Select File'):
    #Show Tkinter dialog box (ask which file you want to open)
    root = tk.Tk()
    root.withdraw()
    filename_csv = askopenfilename(title=windowname,filetypes=[("Dat files","*.dat")])
    column_names = [i for i in range(1,column_number+1,1)]
    vector_num = pd.read_csv(filename_csv, sep=delimiter, skiprows=column_number+1,nrows=1)
    vector_num=int(np.array(vector_num))
    #Skiprows is the line from which the data begins.
    raw_Data= pd.read_csv(filename_csv, sep=delimiter, skiprows=column_number+3+skipped_data,names=column_names )
    Data=[]
    vector_lenght=int(len(raw_Data)/vector_num)
    for i in range(vector_num):#range(len(raw_Data)/vector_lenght):
        pd_vector=raw_Data[i*vector_lenght:(i+1)*vector_lenght]
        Data.append(np.array(pd_vector))
    return Data
