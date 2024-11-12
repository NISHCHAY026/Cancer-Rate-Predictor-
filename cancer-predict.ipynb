{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4cefc5b0",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2024-10-15T08:17:58.685309Z",
     "iopub.status.busy": "2024-10-15T08:17:58.684838Z",
     "iopub.status.idle": "2024-10-15T08:17:59.646063Z",
     "shell.execute_reply": "2024-10-15T08:17:59.644815Z"
    },
    "papermill": {
     "duration": 0.979877,
     "end_time": "2024-10-15T08:17:59.649142",
     "exception": false,
     "start_time": "2024-10-15T08:17:58.669265",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/kaggle/input/esophageal-cancer-dataset/Esophageal_Dataset.csv\n"
     ]
    }
   ],
   "source": [
    "import numpy as np \n",
    "import pandas as pd\n",
    "\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "cc112572",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:17:59.675517Z",
     "iopub.status.busy": "2024-10-15T08:17:59.674765Z",
     "iopub.status.idle": "2024-10-15T08:18:15.497249Z",
     "shell.execute_reply": "2024-10-15T08:18:15.495878Z"
    },
    "papermill": {
     "duration": 15.839501,
     "end_time": "2024-10-15T08:18:15.500826",
     "exception": false,
     "start_time": "2024-10-15T08:17:59.661325",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting wolta\r\n",
      "  Downloading wolta-0.3.3-py3-none-any.whl.metadata (960 bytes)\r\n",
      "Requirement already satisfied: scikit-learn in /opt/conda/lib/python3.10/site-packages (from wolta) (1.2.2)\r\n",
      "Requirement already satisfied: pandas in /opt/conda/lib/python3.10/site-packages (from wolta) (2.2.3)\r\n",
      "Requirement already satisfied: numpy in /opt/conda/lib/python3.10/site-packages (from wolta) (1.26.4)\r\n",
      "Requirement already satisfied: hyperopt in /opt/conda/lib/python3.10/site-packages (from wolta) (0.2.7)\r\n",
      "Requirement already satisfied: catboost in /opt/conda/lib/python3.10/site-packages (from wolta) (1.2.7)\r\n",
      "Collecting imblearn (from wolta)\r\n",
      "  Downloading imblearn-0.0-py2.py3-none-any.whl.metadata (355 bytes)\r\n",
      "Requirement already satisfied: lightgbm in /opt/conda/lib/python3.10/site-packages (from wolta) (4.2.0)\r\n",
      "Requirement already satisfied: matplotlib in /opt/conda/lib/python3.10/site-packages (from wolta) (3.7.5)\r\n",
      "Requirement already satisfied: opencv-python in /opt/conda/lib/python3.10/site-packages (from wolta) (4.10.0.84)\r\n",
      "Requirement already satisfied: graphviz in /opt/conda/lib/python3.10/site-packages (from catboost->wolta) (0.20.3)\r\n",
      "Requirement already satisfied: scipy in /opt/conda/lib/python3.10/site-packages (from catboost->wolta) (1.14.1)\r\n",
      "Requirement already satisfied: plotly in /opt/conda/lib/python3.10/site-packages (from catboost->wolta) (5.22.0)\r\n",
      "Requirement already satisfied: six in /opt/conda/lib/python3.10/site-packages (from catboost->wolta) (1.16.0)\r\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in /opt/conda/lib/python3.10/site-packages (from pandas->wolta) (2.9.0.post0)\r\n",
      "Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.10/site-packages (from pandas->wolta) (2024.1)\r\n",
      "Requirement already satisfied: tzdata>=2022.7 in /opt/conda/lib/python3.10/site-packages (from pandas->wolta) (2024.1)\r\n",
      "Requirement already satisfied: networkx>=2.2 in /opt/conda/lib/python3.10/site-packages (from hyperopt->wolta) (3.3)\r\n",
      "Requirement already satisfied: future in /opt/conda/lib/python3.10/site-packages (from hyperopt->wolta) (1.0.0)\r\n",
      "Requirement already satisfied: tqdm in /opt/conda/lib/python3.10/site-packages (from hyperopt->wolta) (4.66.4)\r\n",
      "Requirement already satisfied: cloudpickle in /opt/conda/lib/python3.10/site-packages (from hyperopt->wolta) (3.0.0)\r\n",
      "Requirement already satisfied: py4j in /opt/conda/lib/python3.10/site-packages (from hyperopt->wolta) (0.10.9.7)\r\n",
      "Requirement already satisfied: imbalanced-learn in /opt/conda/lib/python3.10/site-packages (from imblearn->wolta) (0.12.3)\r\n",
      "Requirement already satisfied: contourpy>=1.0.1 in /opt/conda/lib/python3.10/site-packages (from matplotlib->wolta) (1.2.1)\r\n",
      "Requirement already satisfied: cycler>=0.10 in /opt/conda/lib/python3.10/site-packages (from matplotlib->wolta) (0.12.1)\r\n",
      "Requirement already satisfied: fonttools>=4.22.0 in /opt/conda/lib/python3.10/site-packages (from matplotlib->wolta) (4.53.0)\r\n",
      "Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/lib/python3.10/site-packages (from matplotlib->wolta) (1.4.5)\r\n",
      "Requirement already satisfied: packaging>=20.0 in /opt/conda/lib/python3.10/site-packages (from matplotlib->wolta) (21.3)\r\n",
      "Requirement already satisfied: pillow>=6.2.0 in /opt/conda/lib/python3.10/site-packages (from matplotlib->wolta) (10.3.0)\r\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in /opt/conda/lib/python3.10/site-packages (from matplotlib->wolta) (3.1.2)\r\n",
      "Requirement already satisfied: joblib>=1.1.1 in /opt/conda/lib/python3.10/site-packages (from scikit-learn->wolta) (1.4.2)\r\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/lib/python3.10/site-packages (from scikit-learn->wolta) (3.5.0)\r\n",
      "Requirement already satisfied: tenacity>=6.2.0 in /opt/conda/lib/python3.10/site-packages (from plotly->catboost->wolta) (8.3.0)\r\n",
      "Downloading wolta-0.3.3-py3-none-any.whl (16 kB)\r\n",
      "Downloading imblearn-0.0-py2.py3-none-any.whl (1.9 kB)\r\n",
      "Installing collected packages: imblearn, wolta\r\n",
      "Successfully installed imblearn-0.0 wolta-0.3.3\r\n"
     ]
    }
   ],
   "source": [
    "!pip install wolta"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ff9d342a",
   "metadata": {
    "papermill": {
     "duration": 0.011998,
     "end_time": "2024-10-15T08:18:15.525466",
     "exception": false,
     "start_time": "2024-10-15T08:18:15.513468",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Data Init"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "68aa2252",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:15.553506Z",
     "iopub.status.busy": "2024-10-15T08:18:15.553015Z",
     "iopub.status.idle": "2024-10-15T08:18:15.670649Z",
     "shell.execute_reply": "2024-10-15T08:18:15.669559Z"
    },
    "papermill": {
     "duration": 0.134383,
     "end_time": "2024-10-15T08:18:15.673381",
     "exception": false,
     "start_time": "2024-10-15T08:18:15.538998",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df = pd.read_csv('/kaggle/input/esophageal-cancer-dataset/Esophageal_Dataset.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "acbb9f37",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:15.699988Z",
     "iopub.status.busy": "2024-10-15T08:18:15.699527Z",
     "iopub.status.idle": "2024-10-15T08:18:15.743473Z",
     "shell.execute_reply": "2024-10-15T08:18:15.742381Z"
    },
    "papermill": {
     "duration": 0.06024,
     "end_time": "2024-10-15T08:18:15.745956",
     "exception": false,
     "start_time": "2024-10-15T08:18:15.685716",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>patient_barcode</th>\n",
       "      <th>tissue_source_site</th>\n",
       "      <th>patient_id</th>\n",
       "      <th>bcr_patient_uuid</th>\n",
       "      <th>informed_consent_verified</th>\n",
       "      <th>icd_o_3_site</th>\n",
       "      <th>icd_o_3_histology</th>\n",
       "      <th>icd_10</th>\n",
       "      <th>tissue_prospective_collection_indicator</th>\n",
       "      <th>...</th>\n",
       "      <th>primary_pathology_lymph_node_examined_count</th>\n",
       "      <th>primary_pathology_number_of_lymphnodes_positive_by_he</th>\n",
       "      <th>primary_pathology_number_of_lymphnodes_positive_by_ihc</th>\n",
       "      <th>primary_pathology_planned_surgery_status</th>\n",
       "      <th>primary_pathology_treatment_prior_to_surgery</th>\n",
       "      <th>primary_pathology_residual_tumor</th>\n",
       "      <th>primary_pathology_karnofsky_performance_score</th>\n",
       "      <th>primary_pathology_eastern_cancer_oncology_group</th>\n",
       "      <th>primary_pathology_radiation_therapy</th>\n",
       "      <th>primary_pathology_postoperative_rx_tx</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>TCGA-2H-A9GF</td>\n",
       "      <td>2H</td>\n",
       "      <td>A9GF</td>\n",
       "      <td>0500F1A6-A528-43F3-B035-12D3B7C99C0F</td>\n",
       "      <td>YES</td>\n",
       "      <td>C15.5</td>\n",
       "      <td>8140/3</td>\n",
       "      <td>C15.5</td>\n",
       "      <td>NO</td>\n",
       "      <td>...</td>\n",
       "      <td>8.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>R1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NO</td>\n",
       "      <td>NO</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>TCGA-2H-A9GG</td>\n",
       "      <td>2H</td>\n",
       "      <td>A9GG</td>\n",
       "      <td>70084008-697D-442D-8F74-C12F8F598570</td>\n",
       "      <td>YES</td>\n",
       "      <td>C15.5</td>\n",
       "      <td>8140/3</td>\n",
       "      <td>C15.5</td>\n",
       "      <td>NO</td>\n",
       "      <td>...</td>\n",
       "      <td>19.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>R1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NO</td>\n",
       "      <td>NO</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>TCGA-2H-A9GH</td>\n",
       "      <td>2H</td>\n",
       "      <td>A9GH</td>\n",
       "      <td>606DC5B8-7625-42A6-A936-504EF25623A4</td>\n",
       "      <td>YES</td>\n",
       "      <td>C15.5</td>\n",
       "      <td>8140/3</td>\n",
       "      <td>C15.5</td>\n",
       "      <td>NO</td>\n",
       "      <td>...</td>\n",
       "      <td>30.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>R0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NO</td>\n",
       "      <td>NO</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>TCGA-2H-A9GI</td>\n",
       "      <td>2H</td>\n",
       "      <td>A9GI</td>\n",
       "      <td>CEAF98F8-517E-457A-BF29-ACFE22893D49</td>\n",
       "      <td>YES</td>\n",
       "      <td>C15.5</td>\n",
       "      <td>8140/3</td>\n",
       "      <td>C15.5</td>\n",
       "      <td>NO</td>\n",
       "      <td>...</td>\n",
       "      <td>8.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>R0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NO</td>\n",
       "      <td>NO</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>TCGA-2H-A9GJ</td>\n",
       "      <td>2H</td>\n",
       "      <td>A9GJ</td>\n",
       "      <td>EE47CD59-C8D8-4B1E-96DB-91C679E4106F</td>\n",
       "      <td>YES</td>\n",
       "      <td>C15.5</td>\n",
       "      <td>8140/3</td>\n",
       "      <td>C15.5</td>\n",
       "      <td>NO</td>\n",
       "      <td>...</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>R0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NO</td>\n",
       "      <td>NO</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 85 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0 patient_barcode tissue_source_site patient_id  \\\n",
       "0           0    TCGA-2H-A9GF                 2H       A9GF   \n",
       "1           1    TCGA-2H-A9GG                 2H       A9GG   \n",
       "2           2    TCGA-2H-A9GH                 2H       A9GH   \n",
       "3           3    TCGA-2H-A9GI                 2H       A9GI   \n",
       "4           4    TCGA-2H-A9GJ                 2H       A9GJ   \n",
       "\n",
       "                       bcr_patient_uuid informed_consent_verified  \\\n",
       "0  0500F1A6-A528-43F3-B035-12D3B7C99C0F                       YES   \n",
       "1  70084008-697D-442D-8F74-C12F8F598570                       YES   \n",
       "2  606DC5B8-7625-42A6-A936-504EF25623A4                       YES   \n",
       "3  CEAF98F8-517E-457A-BF29-ACFE22893D49                       YES   \n",
       "4  EE47CD59-C8D8-4B1E-96DB-91C679E4106F                       YES   \n",
       "\n",
       "  icd_o_3_site icd_o_3_histology icd_10  \\\n",
       "0        C15.5            8140/3  C15.5   \n",
       "1        C15.5            8140/3  C15.5   \n",
       "2        C15.5            8140/3  C15.5   \n",
       "3        C15.5            8140/3  C15.5   \n",
       "4        C15.5            8140/3  C15.5   \n",
       "\n",
       "  tissue_prospective_collection_indicator  ...  \\\n",
       "0                                      NO  ...   \n",
       "1                                      NO  ...   \n",
       "2                                      NO  ...   \n",
       "3                                      NO  ...   \n",
       "4                                      NO  ...   \n",
       "\n",
       "  primary_pathology_lymph_node_examined_count  \\\n",
       "0                                         8.0   \n",
       "1                                        19.0   \n",
       "2                                        30.0   \n",
       "3                                         8.0   \n",
       "4                                        19.0   \n",
       "\n",
       "   primary_pathology_number_of_lymphnodes_positive_by_he  \\\n",
       "0                                                7.0       \n",
       "1                                                4.0       \n",
       "2                                                1.0       \n",
       "3                                                4.0       \n",
       "4                                                0.0       \n",
       "\n",
       "  primary_pathology_number_of_lymphnodes_positive_by_ihc  \\\n",
       "0                                                0.0       \n",
       "1                                                0.0       \n",
       "2                                                0.0       \n",
       "3                                                0.0       \n",
       "4                                                0.0       \n",
       "\n",
       "  primary_pathology_planned_surgery_status  \\\n",
       "0                                      NaN   \n",
       "1                                      NaN   \n",
       "2                                      NaN   \n",
       "3                                      NaN   \n",
       "4                                      NaN   \n",
       "\n",
       "   primary_pathology_treatment_prior_to_surgery  \\\n",
       "0                                           NaN   \n",
       "1                                           NaN   \n",
       "2                                           NaN   \n",
       "3                                           NaN   \n",
       "4                                           NaN   \n",
       "\n",
       "   primary_pathology_residual_tumor  \\\n",
       "0                                R1   \n",
       "1                                R1   \n",
       "2                                R0   \n",
       "3                                R0   \n",
       "4                                R0   \n",
       "\n",
       "  primary_pathology_karnofsky_performance_score  \\\n",
       "0                                           NaN   \n",
       "1                                           NaN   \n",
       "2                                           NaN   \n",
       "3                                           NaN   \n",
       "4                                           NaN   \n",
       "\n",
       "  primary_pathology_eastern_cancer_oncology_group  \\\n",
       "0                                             NaN   \n",
       "1                                             NaN   \n",
       "2                                             NaN   \n",
       "3                                             NaN   \n",
       "4                                             NaN   \n",
       "\n",
       "  primary_pathology_radiation_therapy primary_pathology_postoperative_rx_tx  \n",
       "0                                  NO                                    NO  \n",
       "1                                  NO                                    NO  \n",
       "2                                  NO                                    NO  \n",
       "3                                  NO                                    NO  \n",
       "4                                  NO                                    NO  \n",
       "\n",
       "[5 rows x 85 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e9b0d02b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:15.773638Z",
     "iopub.status.busy": "2024-10-15T08:18:15.773195Z",
     "iopub.status.idle": "2024-10-15T08:18:15.780352Z",
     "shell.execute_reply": "2024-10-15T08:18:15.779348Z"
    },
    "papermill": {
     "duration": 0.02398,
     "end_time": "2024-10-15T08:18:15.782924",
     "exception": false,
     "start_time": "2024-10-15T08:18:15.758944",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3985, 85)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "021f567a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:15.810654Z",
     "iopub.status.busy": "2024-10-15T08:18:15.810199Z",
     "iopub.status.idle": "2024-10-15T08:18:15.824429Z",
     "shell.execute_reply": "2024-10-15T08:18:15.823242Z"
    },
    "papermill": {
     "duration": 0.031778,
     "end_time": "2024-10-15T08:18:15.827791",
     "exception": false,
     "start_time": "2024-10-15T08:18:15.796013",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Unnamed: 0: int64\n",
      "patient_barcode: str\n",
      "tissue_source_site: str\n",
      "patient_id: str\n",
      "bcr_patient_uuid: str\n",
      "informed_consent_verified: str\n",
      "icd_o_3_site: str\n",
      "icd_o_3_histology: str\n",
      "icd_10: str\n",
      "tissue_prospective_collection_indicator: str\n",
      "tissue_retrospective_collection_indicator: str\n",
      "days_to_birth: int64\n",
      "country_of_birth: float\n",
      "gender: str\n",
      "height: float64\n",
      "weight: float64\n",
      "country_of_procurement: str\n",
      "state_province_of_procurement: str\n",
      "city_of_procurement: str\n",
      "race_list: float\n",
      "ethnicity: float\n",
      "other_dx: str\n",
      "history_of_neoadjuvant_treatment: str\n",
      "person_neoplasm_cancer_status: str\n",
      "vital_status: str\n",
      "days_to_last_followup: float64\n",
      "days_to_death: float64\n",
      "tobacco_smoking_history: float64\n",
      "age_began_smoking_in_years: float64\n",
      "stopped_smoking_year: float64\n",
      "number_pack_years_smoked: float64\n",
      "alcohol_history_documented: str\n",
      "frequency_of_alcohol_consumption: float64\n",
      "amount_of_alcohol_consumption_per_day: float64\n",
      "reflux_history: float\n",
      "antireflux_treatment_types: float\n",
      "h_pylori_infection: float\n",
      "initial_diagnosis_by: str\n",
      "barretts_esophagus: str\n",
      "goblet_cells_present: float\n",
      "history_of_esophageal_cancer: float\n",
      "number_of_relatives_diagnosed: float64\n",
      "has_new_tumor_events_information: str\n",
      "day_of_form_completion: int64\n",
      "month_of_form_completion: int64\n",
      "year_of_form_completion: int64\n",
      "has_follow_ups_information: str\n",
      "has_drugs_information: str\n",
      "has_radiations_information: str\n",
      "project: str\n",
      "stage_event_system_version: str\n",
      "stage_event_clinical_stage: float\n",
      "stage_event_pathologic_stage: str\n",
      "stage_event_tnm_categories: str\n",
      "stage_event_psa: float64\n",
      "stage_event_gleason_grading: float64\n",
      "stage_event_ann_arbor: float64\n",
      "stage_event_serum_markers: float64\n",
      "stage_event_igcccg_stage: float64\n",
      "stage_event_masaoka_stage: float64\n",
      "primary_pathology_tumor_tissue_site: str\n",
      "primary_pathology_esophageal_tumor_cental_location: str\n",
      "primary_pathology_esophageal_tumor_involvement_sites: str\n",
      "primary_pathology_histological_type: str\n",
      "primary_pathology_columnar_metaplasia_present: str\n",
      "primary_pathology_columnar_mucosa_goblet_cell_present: str\n",
      "primary_pathology_columnar_mucosa_dysplasia: str\n",
      "primary_pathology_neoplasm_histologic_grade: str\n",
      "primary_pathology_days_to_initial_pathologic_diagnosis: int64\n",
      "primary_pathology_age_at_initial_pathologic_diagnosis: int64\n",
      "primary_pathology_year_of_initial_pathologic_diagnosis: float64\n",
      "primary_pathology_initial_pathologic_diagnosis_method: str\n",
      "primary_pathology_init_pathology_dx_method_other: str\n",
      "primary_pathology_lymph_node_metastasis_radiographic_evidence: str\n",
      "primary_pathology_primary_lymph_node_presentation_assessment: str\n",
      "primary_pathology_lymph_node_examined_count: float64\n",
      "primary_pathology_number_of_lymphnodes_positive_by_he: float64\n",
      "primary_pathology_number_of_lymphnodes_positive_by_ihc: float64\n",
      "primary_pathology_planned_surgery_status: float\n",
      "primary_pathology_treatment_prior_to_surgery: float\n",
      "primary_pathology_residual_tumor: str\n",
      "primary_pathology_karnofsky_performance_score: float64\n",
      "primary_pathology_eastern_cancer_oncology_group: float64\n",
      "primary_pathology_radiation_therapy: str\n",
      "primary_pathology_postoperative_rx_tx: str\n"
     ]
    }
   ],
   "source": [
    "from wolta.data_tools import col_types\n",
    "\n",
    "types = col_types(df, print_columns=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ef5408bc",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:15.858351Z",
     "iopub.status.busy": "2024-10-15T08:18:15.857435Z",
     "iopub.status.idle": "2024-10-15T08:18:15.920156Z",
     "shell.execute_reply": "2024-10-15T08:18:15.919008Z"
    },
    "papermill": {
     "duration": 0.08048,
     "end_time": "2024-10-15T08:18:15.923271",
     "exception": false,
     "start_time": "2024-10-15T08:18:15.842791",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tissue_prospective_collection_indicator has 40 null values\n",
      "tissue_retrospective_collection_indicator has 40 null values\n",
      "country_of_birth has 1927 null values\n",
      "height has 219 null values\n",
      "weight has 40 null values\n",
      "country_of_procurement has 40 null values\n",
      "state_province_of_procurement has 1280 null values\n",
      "city_of_procurement has 860 null values\n",
      "race_list has 419 null values\n",
      "ethnicity has 2048 null values\n",
      "person_neoplasm_cancer_status has 335 null values\n",
      "days_to_last_followup has 1197 null values\n",
      "days_to_death has 2788 null values\n",
      "tobacco_smoking_history has 380 null values\n",
      "age_began_smoking_in_years has 2253 null values\n",
      "stopped_smoking_year has 2377 null values\n",
      "number_pack_years_smoked has 1816 null values\n",
      "alcohol_history_documented has 60 null values\n",
      "frequency_of_alcohol_consumption has 1516 null values\n",
      "amount_of_alcohol_consumption_per_day has 1877 null values\n",
      "reflux_history has 677 null values\n",
      "antireflux_treatment_types has 2988 null values\n",
      "h_pylori_infection has 2428 null values\n",
      "initial_diagnosis_by has 738 null values\n",
      "barretts_esophagus has 818 null values\n",
      "goblet_cells_present has 3566 null values\n",
      "history_of_esophageal_cancer has 837 null values\n",
      "number_of_relatives_diagnosed has 3146 null values\n",
      "stage_event_clinical_stage has 2667 null values\n",
      "stage_event_pathologic_stage has 498 null values\n",
      "stage_event_psa has 3985 null values\n",
      "stage_event_gleason_grading has 3985 null values\n",
      "stage_event_ann_arbor has 3985 null values\n",
      "stage_event_serum_markers has 3985 null values\n",
      "stage_event_igcccg_stage has 3985 null values\n",
      "stage_event_masaoka_stage has 3985 null values\n",
      "primary_pathology_esophageal_tumor_cental_location has 20 null values\n",
      "primary_pathology_esophageal_tumor_involvement_sites has 20 null values\n",
      "primary_pathology_columnar_metaplasia_present has 1595 null values\n",
      "primary_pathology_columnar_mucosa_goblet_cell_present has 2174 null values\n",
      "primary_pathology_columnar_mucosa_dysplasia has 2235 null values\n",
      "primary_pathology_year_of_initial_pathologic_diagnosis has 140 null values\n",
      "primary_pathology_initial_pathologic_diagnosis_method has 100 null values\n",
      "primary_pathology_init_pathology_dx_method_other has 3106 null values\n",
      "primary_pathology_lymph_node_metastasis_radiographic_evidence has 837 null values\n",
      "primary_pathology_primary_lymph_node_presentation_assessment has 320 null values\n",
      "primary_pathology_lymph_node_examined_count has 1000 null values\n",
      "primary_pathology_number_of_lymphnodes_positive_by_he has 1000 null values\n",
      "primary_pathology_number_of_lymphnodes_positive_by_ihc has 2533 null values\n",
      "primary_pathology_planned_surgery_status has 2507 null values\n",
      "primary_pathology_treatment_prior_to_surgery has 2847 null values\n",
      "primary_pathology_residual_tumor has 520 null values\n",
      "primary_pathology_karnofsky_performance_score has 2625 null values\n",
      "primary_pathology_eastern_cancer_oncology_group has 2628 null values\n",
      "primary_pathology_radiation_therapy has 638 null values\n",
      "primary_pathology_postoperative_rx_tx has 658 null values\n"
     ]
    }
   ],
   "source": [
    "from wolta.data_tools import seek_null\n",
    "\n",
    "seeked = seek_null(df, print_columns=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "4c0e2d38",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:15.952253Z",
     "iopub.status.busy": "2024-10-15T08:18:15.951314Z",
     "iopub.status.idle": "2024-10-15T08:18:15.987349Z",
     "shell.execute_reply": "2024-10-15T08:18:15.986295Z"
    },
    "papermill": {
     "duration": 0.053125,
     "end_time": "2024-10-15T08:18:15.989821",
     "exception": false,
     "start_time": "2024-10-15T08:18:15.936696",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'Unnamed: 0': 3985,\n",
       " 'patient_barcode': 3985,\n",
       " 'tissue_source_site': 19,\n",
       " 'patient_id': 185,\n",
       " 'bcr_patient_uuid': 185,\n",
       " 'informed_consent_verified': 1,\n",
       " 'icd_o_3_site': 6,\n",
       " 'icd_o_3_histology': 6,\n",
       " 'icd_10': 5,\n",
       " 'tissue_prospective_collection_indicator': 3,\n",
       " 'tissue_retrospective_collection_indicator': 3,\n",
       " 'days_to_birth': 3344,\n",
       " 'country_of_birth': 9,\n",
       " 'gender': 2,\n",
       " 'height': 58,\n",
       " 'weight': 109,\n",
       " 'country_of_procurement': 11,\n",
       " 'state_province_of_procurement': 21,\n",
       " 'city_of_procurement': 29,\n",
       " 'race_list': 4,\n",
       " 'ethnicity': 3,\n",
       " 'other_dx': 2,\n",
       " 'history_of_neoadjuvant_treatment': 1,\n",
       " 'person_neoplasm_cancer_status': 3,\n",
       " 'vital_status': 2,\n",
       " 'days_to_last_followup': 85,\n",
       " 'days_to_death': 58,\n",
       " 'tobacco_smoking_history': 5,\n",
       " 'age_began_smoking_in_years': 28,\n",
       " 'stopped_smoking_year': 40,\n",
       " 'number_pack_years_smoked': 47,\n",
       " 'alcohol_history_documented': 3,\n",
       " 'frequency_of_alcohol_consumption': 8,\n",
       " 'amount_of_alcohol_consumption_per_day': 12,\n",
       " 'reflux_history': 3,\n",
       " 'antireflux_treatment_types': 4,\n",
       " 'h_pylori_infection': 3,\n",
       " 'initial_diagnosis_by': 4,\n",
       " 'barretts_esophagus': 4,\n",
       " 'goblet_cells_present': 3,\n",
       " 'history_of_esophageal_cancer': 3,\n",
       " 'number_of_relatives_diagnosed': 4,\n",
       " 'has_new_tumor_events_information': 2,\n",
       " 'day_of_form_completion': 22,\n",
       " 'month_of_form_completion': 12,\n",
       " 'year_of_form_completion': 4,\n",
       " 'has_follow_ups_information': 2,\n",
       " 'has_drugs_information': 2,\n",
       " 'has_radiations_information': 2,\n",
       " 'project': 1,\n",
       " 'stage_event_system_version': 3,\n",
       " 'stage_event_clinical_stage': 13,\n",
       " 'stage_event_pathologic_stage': 13,\n",
       " 'stage_event_tnm_categories': 54,\n",
       " 'stage_event_psa': 1,\n",
       " 'stage_event_gleason_grading': 1,\n",
       " 'stage_event_ann_arbor': 1,\n",
       " 'stage_event_serum_markers': 1,\n",
       " 'stage_event_igcccg_stage': 1,\n",
       " 'stage_event_masaoka_stage': 1,\n",
       " 'primary_pathology_tumor_tissue_site': 1,\n",
       " 'primary_pathology_esophageal_tumor_cental_location': 4,\n",
       " 'primary_pathology_esophageal_tumor_involvement_sites': 6,\n",
       " 'primary_pathology_histological_type': 2,\n",
       " 'primary_pathology_columnar_metaplasia_present': 3,\n",
       " 'primary_pathology_columnar_mucosa_goblet_cell_present': 3,\n",
       " 'primary_pathology_columnar_mucosa_dysplasia': 4,\n",
       " 'primary_pathology_neoplasm_histologic_grade': 4,\n",
       " 'primary_pathology_days_to_initial_pathologic_diagnosis': 1,\n",
       " 'primary_pathology_age_at_initial_pathologic_diagnosis': 46,\n",
       " 'primary_pathology_year_of_initial_pathologic_diagnosis': 16,\n",
       " 'primary_pathology_initial_pathologic_diagnosis_method': 4,\n",
       " 'primary_pathology_init_pathology_dx_method_other': 11,\n",
       " 'primary_pathology_lymph_node_metastasis_radiographic_evidence': 3,\n",
       " 'primary_pathology_primary_lymph_node_presentation_assessment': 3,\n",
       " 'primary_pathology_lymph_node_examined_count': 40,\n",
       " 'primary_pathology_number_of_lymphnodes_positive_by_he': 15,\n",
       " 'primary_pathology_number_of_lymphnodes_positive_by_ihc': 6,\n",
       " 'primary_pathology_planned_surgery_status': 4,\n",
       " 'primary_pathology_treatment_prior_to_surgery': 3,\n",
       " 'primary_pathology_residual_tumor': 5,\n",
       " 'primary_pathology_karnofsky_performance_score': 9,\n",
       " 'primary_pathology_eastern_cancer_oncology_group': 5,\n",
       " 'primary_pathology_radiation_therapy': 3,\n",
       " 'primary_pathology_postoperative_rx_tx': 3}"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from wolta.data_tools import unique_amounts\n",
    "\n",
    "unique_amounts(df)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2338dc7e",
   "metadata": {
    "papermill": {
     "duration": 0.013405,
     "end_time": "2024-10-15T08:18:16.016973",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.003568",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Data Manipulation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a8895077",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:16.046514Z",
     "iopub.status.busy": "2024-10-15T08:18:16.046095Z",
     "iopub.status.idle": "2024-10-15T08:18:16.051655Z",
     "shell.execute_reply": "2024-10-15T08:18:16.050611Z"
    },
    "papermill": {
     "duration": 0.022993,
     "end_time": "2024-10-15T08:18:16.054135",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.031142",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "will_del = ['Unnamed: 0',\n",
    "           'patient_barcode',\n",
    "           'tissue_source_site',\n",
    "           'patient_id',\n",
    "           'bcr_patient_uuid',\n",
    "           'informed_consent_verified',\n",
    "           'icd_o_3_site',\n",
    "           'icd_o_3_histology',\n",
    "           'icd_10',\n",
    "           'tissue_prospective_collection_indicator',\n",
    "           'tissue_retrospective_collection_indicator']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "9bfd0577",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:16.083869Z",
     "iopub.status.busy": "2024-10-15T08:18:16.083404Z",
     "iopub.status.idle": "2024-10-15T08:18:16.162702Z",
     "shell.execute_reply": "2024-10-15T08:18:16.161533Z"
    },
    "papermill": {
     "duration": 0.097323,
     "end_time": "2024-10-15T08:18:16.165788",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.068465",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The maximum tolerated null value amount is 398\n",
      "country_of_birth will be deleted because it has 1927 null values and this is 1529 values more than tolerance\n",
      "state_province_of_procurement will be deleted because it has 1280 null values and this is 882 values more than tolerance\n",
      "city_of_procurement will be deleted because it has 860 null values and this is 462 values more than tolerance\n",
      "race_list will be deleted because it has 419 null values and this is 21 values more than tolerance\n",
      "ethnicity will be deleted because it has 2048 null values and this is 1650 values more than tolerance\n",
      "days_to_last_followup will be deleted because it has 1197 null values and this is 799 values more than tolerance\n",
      "days_to_death will be deleted because it has 2788 null values and this is 2390 values more than tolerance\n",
      "age_began_smoking_in_years will be deleted because it has 2253 null values and this is 1855 values more than tolerance\n",
      "stopped_smoking_year will be deleted because it has 2377 null values and this is 1979 values more than tolerance\n",
      "number_pack_years_smoked will be deleted because it has 1816 null values and this is 1418 values more than tolerance\n",
      "frequency_of_alcohol_consumption will be deleted because it has 1516 null values and this is 1118 values more than tolerance\n",
      "amount_of_alcohol_consumption_per_day will be deleted because it has 1877 null values and this is 1479 values more than tolerance\n",
      "reflux_history will be deleted because it has 677 null values and this is 279 values more than tolerance\n",
      "antireflux_treatment_types will be deleted because it has 2988 null values and this is 2590 values more than tolerance\n",
      "h_pylori_infection will be deleted because it has 2428 null values and this is 2030 values more than tolerance\n",
      "initial_diagnosis_by will be deleted because it has 738 null values and this is 340 values more than tolerance\n",
      "barretts_esophagus will be deleted because it has 818 null values and this is 420 values more than tolerance\n",
      "goblet_cells_present will be deleted because it has 3566 null values and this is 3168 values more than tolerance\n",
      "history_of_esophageal_cancer will be deleted because it has 837 null values and this is 439 values more than tolerance\n",
      "number_of_relatives_diagnosed will be deleted because it has 3146 null values and this is 2748 values more than tolerance\n",
      "stage_event_clinical_stage will be deleted because it has 2667 null values and this is 2269 values more than tolerance\n",
      "stage_event_pathologic_stage will be deleted because it has 498 null values and this is 100 values more than tolerance\n",
      "stage_event_psa will be deleted because it has 3985 null values and this is 3587 values more than tolerance\n",
      "stage_event_gleason_grading will be deleted because it has 3985 null values and this is 3587 values more than tolerance\n",
      "stage_event_ann_arbor will be deleted because it has 3985 null values and this is 3587 values more than tolerance\n",
      "stage_event_serum_markers will be deleted because it has 3985 null values and this is 3587 values more than tolerance\n",
      "stage_event_igcccg_stage will be deleted because it has 3985 null values and this is 3587 values more than tolerance\n",
      "stage_event_masaoka_stage will be deleted because it has 3985 null values and this is 3587 values more than tolerance\n",
      "primary_pathology_columnar_metaplasia_present will be deleted because it has 1595 null values and this is 1197 values more than tolerance\n",
      "primary_pathology_columnar_mucosa_goblet_cell_present will be deleted because it has 2174 null values and this is 1776 values more than tolerance\n",
      "primary_pathology_columnar_mucosa_dysplasia will be deleted because it has 2235 null values and this is 1837 values more than tolerance\n",
      "primary_pathology_init_pathology_dx_method_other will be deleted because it has 3106 null values and this is 2708 values more than tolerance\n",
      "primary_pathology_lymph_node_metastasis_radiographic_evidence will be deleted because it has 837 null values and this is 439 values more than tolerance\n",
      "primary_pathology_lymph_node_examined_count will be deleted because it has 1000 null values and this is 602 values more than tolerance\n",
      "primary_pathology_number_of_lymphnodes_positive_by_he will be deleted because it has 1000 null values and this is 602 values more than tolerance\n",
      "primary_pathology_number_of_lymphnodes_positive_by_ihc will be deleted because it has 2533 null values and this is 2135 values more than tolerance\n",
      "primary_pathology_planned_surgery_status will be deleted because it has 2507 null values and this is 2109 values more than tolerance\n",
      "primary_pathology_treatment_prior_to_surgery will be deleted because it has 2847 null values and this is 2449 values more than tolerance\n",
      "primary_pathology_residual_tumor will be deleted because it has 520 null values and this is 122 values more than tolerance\n",
      "primary_pathology_karnofsky_performance_score will be deleted because it has 2625 null values and this is 2227 values more than tolerance\n",
      "primary_pathology_eastern_cancer_oncology_group will be deleted because it has 2628 null values and this is 2230 values more than tolerance\n",
      "primary_pathology_radiation_therapy will be deleted because it has 638 null values and this is 240 values more than tolerance\n",
      "primary_pathology_postoperative_rx_tx will be deleted because it has 658 null values and this is 260 values more than tolerance\n",
      "history_of_neoadjuvant_treatment will be deleted because it has single value\n",
      "project will be deleted because it has single value\n",
      "primary_pathology_tumor_tissue_site will be deleted because it has single value\n",
      "primary_pathology_days_to_initial_pathologic_diagnosis will be deleted because it has single value\n",
      "The maximum tolerated unique value amount is 398 in string data\n"
     ]
    }
   ],
   "source": [
    "from wolta.feature_tools import list_deletings\n",
    "\n",
    "df = list_deletings(df, extra=will_del, null_tolerance=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "362b7f61",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:16.195524Z",
     "iopub.status.busy": "2024-10-15T08:18:16.195097Z",
     "iopub.status.idle": "2024-10-15T08:18:16.219068Z",
     "shell.execute_reply": "2024-10-15T08:18:16.217772Z"
    },
    "papermill": {
     "duration": 0.041862,
     "end_time": "2024-10-15T08:18:16.221614",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.179752",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "height has 219 null values\n",
      "weight has 40 null values\n",
      "country_of_procurement has 40 null values\n",
      "person_neoplasm_cancer_status has 335 null values\n",
      "tobacco_smoking_history has 380 null values\n",
      "alcohol_history_documented has 60 null values\n",
      "primary_pathology_esophageal_tumor_cental_location has 20 null values\n",
      "primary_pathology_esophageal_tumor_involvement_sites has 20 null values\n",
      "primary_pathology_year_of_initial_pathologic_diagnosis has 140 null values\n",
      "primary_pathology_initial_pathologic_diagnosis_method has 100 null values\n",
      "primary_pathology_primary_lymph_node_presentation_assessment has 320 null values\n"
     ]
    }
   ],
   "source": [
    "seeked = seek_null(df, print_columns=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "490097e7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:16.251614Z",
     "iopub.status.busy": "2024-10-15T08:18:16.251142Z",
     "iopub.status.idle": "2024-10-15T08:18:16.263628Z",
     "shell.execute_reply": "2024-10-15T08:18:16.262617Z"
    },
    "papermill": {
     "duration": 0.030655,
     "end_time": "2024-10-15T08:18:16.266379",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.235724",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df['person_neoplasm_cancer_status'] = df['person_neoplasm_cancer_status'].fillna('WILLBEDELETED')\n",
    "df = df[df['person_neoplasm_cancer_status'] != 'WILLBEDELETED']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "1e212842",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:16.296070Z",
     "iopub.status.busy": "2024-10-15T08:18:16.295587Z",
     "iopub.status.idle": "2024-10-15T08:18:16.302660Z",
     "shell.execute_reply": "2024-10-15T08:18:16.301574Z"
    },
    "papermill": {
     "duration": 0.024584,
     "end_time": "2024-10-15T08:18:16.305001",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.280417",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3650, 27)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "7e9cbf69",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:16.334529Z",
     "iopub.status.busy": "2024-10-15T08:18:16.334085Z",
     "iopub.status.idle": "2024-10-15T08:18:16.341382Z",
     "shell.execute_reply": "2024-10-15T08:18:16.340279Z"
    },
    "papermill": {
     "duration": 0.025723,
     "end_time": "2024-10-15T08:18:16.344565",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.318842",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "days_to_birth: int64\n",
      "gender: str\n",
      "height: float64\n",
      "weight: float64\n",
      "country_of_procurement: str\n",
      "other_dx: str\n",
      "person_neoplasm_cancer_status: str\n",
      "vital_status: str\n",
      "tobacco_smoking_history: float64\n",
      "alcohol_history_documented: str\n",
      "has_new_tumor_events_information: str\n",
      "day_of_form_completion: int64\n",
      "month_of_form_completion: int64\n",
      "year_of_form_completion: int64\n",
      "has_follow_ups_information: str\n",
      "has_drugs_information: str\n",
      "has_radiations_information: str\n",
      "stage_event_system_version: str\n",
      "stage_event_tnm_categories: str\n",
      "primary_pathology_esophageal_tumor_cental_location: str\n",
      "primary_pathology_esophageal_tumor_involvement_sites: str\n",
      "primary_pathology_histological_type: str\n",
      "primary_pathology_neoplasm_histologic_grade: str\n",
      "primary_pathology_age_at_initial_pathologic_diagnosis: int64\n",
      "primary_pathology_year_of_initial_pathologic_diagnosis: float64\n",
      "primary_pathology_initial_pathologic_diagnosis_method: str\n",
      "primary_pathology_primary_lymph_node_presentation_assessment: str\n"
     ]
    }
   ],
   "source": [
    "types = col_types(df, print_columns=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "acf74b06",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:16.374592Z",
     "iopub.status.busy": "2024-10-15T08:18:16.374163Z",
     "iopub.status.idle": "2024-10-15T08:18:16.396425Z",
     "shell.execute_reply": "2024-10-15T08:18:16.395401Z"
    },
    "papermill": {
     "duration": 0.040943,
     "end_time": "2024-10-15T08:18:16.399659",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.358716",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "height has 160 null values\n",
      "weight has 20 null values\n",
      "country_of_procurement has 40 null values\n",
      "tobacco_smoking_history has 360 null values\n",
      "alcohol_history_documented has 60 null values\n",
      "primary_pathology_esophageal_tumor_cental_location has 20 null values\n",
      "primary_pathology_esophageal_tumor_involvement_sites has 20 null values\n",
      "primary_pathology_year_of_initial_pathologic_diagnosis has 140 null values\n",
      "primary_pathology_initial_pathologic_diagnosis_method has 100 null values\n",
      "primary_pathology_primary_lymph_node_presentation_assessment has 300 null values\n"
     ]
    }
   ],
   "source": [
    "seeked = seek_null(df, print_columns=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "68e5e0c7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:16.430397Z",
     "iopub.status.busy": "2024-10-15T08:18:16.429973Z",
     "iopub.status.idle": "2024-10-15T08:18:16.438660Z",
     "shell.execute_reply": "2024-10-15T08:18:16.437725Z"
    },
    "papermill": {
     "duration": 0.02677,
     "end_time": "2024-10-15T08:18:16.441014",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.414244",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df['height'] = df['height'].fillna(np.nanmean(df['height'].values))\n",
    "df['weight'] = df['weight'].fillna(np.nanmean(df['weight'].values))\n",
    "df['tobacco_smoking_history'] = df['tobacco_smoking_history'].fillna(np.nanmean(df['tobacco_smoking_history'].values))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "9d6d8c3f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:16.471410Z",
     "iopub.status.busy": "2024-10-15T08:18:16.470970Z",
     "iopub.status.idle": "2024-10-15T08:18:16.480571Z",
     "shell.execute_reply": "2024-10-15T08:18:16.479510Z"
    },
    "papermill": {
     "duration": 0.02763,
     "end_time": "2024-10-15T08:18:16.482951",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.455321",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df['primary_pathology_year_of_initial_pathologic_diagnosis'] = df['primary_pathology_year_of_initial_pathologic_diagnosis'].astype(str)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "68b661f3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:16.513760Z",
     "iopub.status.busy": "2024-10-15T08:18:16.513003Z",
     "iopub.status.idle": "2024-10-15T08:18:16.529060Z",
     "shell.execute_reply": "2024-10-15T08:18:16.527856Z"
    },
    "papermill": {
     "duration": 0.034177,
     "end_time": "2024-10-15T08:18:16.531589",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.497412",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df['country_of_procurement'] = df['country_of_procurement'].fillna('UNKNOWN')\n",
    "df['alcohol_history_documented'] = df['alcohol_history_documented'].fillna('UNKNOWN')\n",
    "df['primary_pathology_esophageal_tumor_cental_location'] = df['primary_pathology_esophageal_tumor_cental_location'].fillna('UNKNOWN')\n",
    "df['primary_pathology_esophageal_tumor_involvement_sites'] = df['primary_pathology_esophageal_tumor_involvement_sites'].fillna('UNKNOWN')\n",
    "df['primary_pathology_year_of_initial_pathologic_diagnosis'] = df['primary_pathology_year_of_initial_pathologic_diagnosis'].fillna('UNKNOWN')\n",
    "df['primary_pathology_initial_pathologic_diagnosis_method'] = df['primary_pathology_initial_pathologic_diagnosis_method'].fillna('UNKNOWN')\n",
    "df['primary_pathology_primary_lymph_node_presentation_assessment'] = df['primary_pathology_primary_lymph_node_presentation_assessment'].fillna('UKNOWN')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "51f2def3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:16.562323Z",
     "iopub.status.busy": "2024-10-15T08:18:16.561492Z",
     "iopub.status.idle": "2024-10-15T08:18:16.569746Z",
     "shell.execute_reply": "2024-10-15T08:18:16.568773Z"
    },
    "papermill": {
     "duration": 0.026738,
     "end_time": "2024-10-15T08:18:16.572667",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.545929",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'WITH TUMOR': 0, 'TUMOR FREE': 1}\n",
      "['WITH TUMOR', 'TUMOR FREE']\n"
     ]
    }
   ],
   "source": [
    "from wolta.data_tools import make_numerics\n",
    "\n",
    "df['person_neoplasm_cancer_status'], outs = make_numerics(df['person_neoplasm_cancer_status'], space_requested=True)\n",
    "\n",
    "print(outs)\n",
    "outs = list(outs)\n",
    "print(outs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "197f0d9e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:16.603122Z",
     "iopub.status.busy": "2024-10-15T08:18:16.602694Z",
     "iopub.status.idle": "2024-10-15T08:18:16.633817Z",
     "shell.execute_reply": "2024-10-15T08:18:16.632794Z"
    },
    "papermill": {
     "duration": 0.049706,
     "end_time": "2024-10-15T08:18:16.636593",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.586887",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "types = col_types(df)\n",
    "loc = 0\n",
    "\n",
    "for col in df.columns:\n",
    "    if types[loc] == 'str':\n",
    "        df[col] = make_numerics(df[col])\n",
    "    \n",
    "    loc += 1"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1f35d63b",
   "metadata": {
    "papermill": {
     "duration": 0.013928,
     "end_time": "2024-10-15T08:18:16.664821",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.650893",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Data Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "f3015d94",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:16.695894Z",
     "iopub.status.busy": "2024-10-15T08:18:16.694615Z",
     "iopub.status.idle": "2024-10-15T08:18:16.721354Z",
     "shell.execute_reply": "2024-10-15T08:18:16.720139Z"
    },
    "papermill": {
     "duration": 0.044826,
     "end_time": "2024-10-15T08:18:16.723988",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.679162",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>days_to_birth</th>\n",
       "      <th>gender</th>\n",
       "      <th>height</th>\n",
       "      <th>weight</th>\n",
       "      <th>country_of_procurement</th>\n",
       "      <th>other_dx</th>\n",
       "      <th>person_neoplasm_cancer_status</th>\n",
       "      <th>vital_status</th>\n",
       "      <th>tobacco_smoking_history</th>\n",
       "      <th>alcohol_history_documented</th>\n",
       "      <th>...</th>\n",
       "      <th>stage_event_system_version</th>\n",
       "      <th>stage_event_tnm_categories</th>\n",
       "      <th>primary_pathology_esophageal_tumor_cental_location</th>\n",
       "      <th>primary_pathology_esophageal_tumor_involvement_sites</th>\n",
       "      <th>primary_pathology_histological_type</th>\n",
       "      <th>primary_pathology_neoplasm_histologic_grade</th>\n",
       "      <th>primary_pathology_age_at_initial_pathologic_diagnosis</th>\n",
       "      <th>primary_pathology_year_of_initial_pathologic_diagnosis</th>\n",
       "      <th>primary_pathology_initial_pathologic_diagnosis_method</th>\n",
       "      <th>primary_pathology_primary_lymph_node_presentation_assessment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-24487</td>\n",
       "      <td>0</td>\n",
       "      <td>183.0</td>\n",
       "      <td>95.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.32614</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>67</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-24328</td>\n",
       "      <td>0</td>\n",
       "      <td>178.0</td>\n",
       "      <td>74.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.32614</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>66</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-16197</td>\n",
       "      <td>0</td>\n",
       "      <td>183.0</td>\n",
       "      <td>91.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.32614</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>44</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-25097</td>\n",
       "      <td>0</td>\n",
       "      <td>188.0</td>\n",
       "      <td>100.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.32614</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>68</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-21180</td>\n",
       "      <td>0</td>\n",
       "      <td>189.0</td>\n",
       "      <td>70.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.32614</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>57</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 27 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   days_to_birth  gender  height  weight  country_of_procurement  other_dx  \\\n",
       "0         -24487       0   183.0    95.0                       0         0   \n",
       "1         -24328       0   178.0    74.0                       0         0   \n",
       "2         -16197       0   183.0    91.0                       0         0   \n",
       "3         -25097       0   188.0   100.0                       0         0   \n",
       "4         -21180       0   189.0    70.0                       0         0   \n",
       "\n",
       "   person_neoplasm_cancer_status  vital_status  tobacco_smoking_history  \\\n",
       "0                              0             0                  2.32614   \n",
       "1                              0             0                  2.32614   \n",
       "2                              0             0                  2.32614   \n",
       "3                              0             0                  2.32614   \n",
       "4                              0             0                  2.32614   \n",
       "\n",
       "   alcohol_history_documented  ...  stage_event_system_version  \\\n",
       "0                           0  ...                           0   \n",
       "1                           0  ...                           0   \n",
       "2                           0  ...                           0   \n",
       "3                           0  ...                           0   \n",
       "4                           0  ...                           0   \n",
       "\n",
       "   stage_event_tnm_categories  \\\n",
       "0                           0   \n",
       "1                           0   \n",
       "2                           1   \n",
       "3                           0   \n",
       "4                           2   \n",
       "\n",
       "   primary_pathology_esophageal_tumor_cental_location  \\\n",
       "0                                                  0    \n",
       "1                                                  0    \n",
       "2                                                  0    \n",
       "3                                                  0    \n",
       "4                                                  0    \n",
       "\n",
       "   primary_pathology_esophageal_tumor_involvement_sites  \\\n",
       "0                                                  0      \n",
       "1                                                  0      \n",
       "2                                                  0      \n",
       "3                                                  0      \n",
       "4                                                  0      \n",
       "\n",
       "   primary_pathology_histological_type  \\\n",
       "0                                    0   \n",
       "1                                    0   \n",
       "2                                    0   \n",
       "3                                    0   \n",
       "4                                    0   \n",
       "\n",
       "   primary_pathology_neoplasm_histologic_grade  \\\n",
       "0                                            0   \n",
       "1                                            1   \n",
       "2                                            1   \n",
       "3                                            1   \n",
       "4                                            1   \n",
       "\n",
       "   primary_pathology_age_at_initial_pathologic_diagnosis  \\\n",
       "0                                                 67       \n",
       "1                                                 66       \n",
       "2                                                 44       \n",
       "3                                                 68       \n",
       "4                                                 57       \n",
       "\n",
       "   primary_pathology_year_of_initial_pathologic_diagnosis  \\\n",
       "0                                                  0        \n",
       "1                                                  1        \n",
       "2                                                  2        \n",
       "3                                                  1        \n",
       "4                                                  3        \n",
       "\n",
       "   primary_pathology_initial_pathologic_diagnosis_method  \\\n",
       "0                                                  0       \n",
       "1                                                  0       \n",
       "2                                                  0       \n",
       "3                                                  0       \n",
       "4                                                  0       \n",
       "\n",
       "   primary_pathology_primary_lymph_node_presentation_assessment  \n",
       "0                                                  0             \n",
       "1                                                  0             \n",
       "2                                                  0             \n",
       "3                                                  0             \n",
       "4                                                  0             \n",
       "\n",
       "[5 rows x 27 columns]"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "8c1d5523",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:16.755369Z",
     "iopub.status.busy": "2024-10-15T08:18:16.754931Z",
     "iopub.status.idle": "2024-10-15T08:18:16.839222Z",
     "shell.execute_reply": "2024-10-15T08:18:16.837983Z"
    },
    "papermill": {
     "duration": 0.104294,
     "end_time": "2024-10-15T08:18:16.843073",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.738779",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>days_to_birth</th>\n",
       "      <th>gender</th>\n",
       "      <th>height</th>\n",
       "      <th>weight</th>\n",
       "      <th>country_of_procurement</th>\n",
       "      <th>other_dx</th>\n",
       "      <th>person_neoplasm_cancer_status</th>\n",
       "      <th>vital_status</th>\n",
       "      <th>tobacco_smoking_history</th>\n",
       "      <th>alcohol_history_documented</th>\n",
       "      <th>...</th>\n",
       "      <th>stage_event_system_version</th>\n",
       "      <th>stage_event_tnm_categories</th>\n",
       "      <th>primary_pathology_esophageal_tumor_cental_location</th>\n",
       "      <th>primary_pathology_esophageal_tumor_involvement_sites</th>\n",
       "      <th>primary_pathology_histological_type</th>\n",
       "      <th>primary_pathology_neoplasm_histologic_grade</th>\n",
       "      <th>primary_pathology_age_at_initial_pathologic_diagnosis</th>\n",
       "      <th>primary_pathology_year_of_initial_pathologic_diagnosis</th>\n",
       "      <th>primary_pathology_initial_pathologic_diagnosis_method</th>\n",
       "      <th>primary_pathology_primary_lymph_node_presentation_assessment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "      <td>3650.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>-22980.244110</td>\n",
       "      <td>0.147397</td>\n",
       "      <td>172.145559</td>\n",
       "      <td>74.577686</td>\n",
       "      <td>2.980822</td>\n",
       "      <td>0.106849</td>\n",
       "      <td>0.612329</td>\n",
       "      <td>0.672055</td>\n",
       "      <td>2.326140</td>\n",
       "      <td>0.731781</td>\n",
       "      <td>...</td>\n",
       "      <td>1.219452</td>\n",
       "      <td>12.667671</td>\n",
       "      <td>0.355616</td>\n",
       "      <td>0.476164</td>\n",
       "      <td>0.530685</td>\n",
       "      <td>1.158082</td>\n",
       "      <td>62.428493</td>\n",
       "      <td>6.567945</td>\n",
       "      <td>1.021370</td>\n",
       "      <td>0.443836</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>4348.093538</td>\n",
       "      <td>0.354550</td>\n",
       "      <td>8.840043</td>\n",
       "      <td>17.315773</td>\n",
       "      <td>2.477006</td>\n",
       "      <td>0.308964</td>\n",
       "      <td>0.487286</td>\n",
       "      <td>0.469529</td>\n",
       "      <td>1.097476</td>\n",
       "      <td>0.478766</td>\n",
       "      <td>...</td>\n",
       "      <td>0.674725</td>\n",
       "      <td>12.582748</td>\n",
       "      <td>0.572602</td>\n",
       "      <td>0.849036</td>\n",
       "      <td>0.499126</td>\n",
       "      <td>0.924942</td>\n",
       "      <td>11.937696</td>\n",
       "      <td>3.972903</td>\n",
       "      <td>0.709396</td>\n",
       "      <td>0.780164</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>-32972.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>145.000000</td>\n",
       "      <td>41.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>27.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>-26347.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>167.000000</td>\n",
       "      <td>62.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>54.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>-22163.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>172.145559</td>\n",
       "      <td>72.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>60.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>-19818.250000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>178.000000</td>\n",
       "      <td>85.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>72.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>-10143.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>202.000000</td>\n",
       "      <td>138.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>48.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>90.000000</td>\n",
       "      <td>15.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>8 rows × 27 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       days_to_birth       gender       height       weight  \\\n",
       "count    3650.000000  3650.000000  3650.000000  3650.000000   \n",
       "mean   -22980.244110     0.147397   172.145559    74.577686   \n",
       "std      4348.093538     0.354550     8.840043    17.315773   \n",
       "min    -32972.000000     0.000000   145.000000    41.000000   \n",
       "25%    -26347.500000     0.000000   167.000000    62.000000   \n",
       "50%    -22163.500000     0.000000   172.145559    72.000000   \n",
       "75%    -19818.250000     0.000000   178.000000    85.000000   \n",
       "max    -10143.000000     1.000000   202.000000   138.000000   \n",
       "\n",
       "       country_of_procurement     other_dx  person_neoplasm_cancer_status  \\\n",
       "count             3650.000000  3650.000000                    3650.000000   \n",
       "mean                 2.980822     0.106849                       0.612329   \n",
       "std                  2.477006     0.308964                       0.487286   \n",
       "min                  0.000000     0.000000                       0.000000   \n",
       "25%                  1.000000     0.000000                       0.000000   \n",
       "50%                  2.000000     0.000000                       1.000000   \n",
       "75%                  4.000000     0.000000                       1.000000   \n",
       "max                  9.000000     1.000000                       1.000000   \n",
       "\n",
       "       vital_status  tobacco_smoking_history  alcohol_history_documented  ...  \\\n",
       "count   3650.000000              3650.000000                 3650.000000  ...   \n",
       "mean       0.672055                 2.326140                    0.731781  ...   \n",
       "std        0.469529                 1.097476                    0.478766  ...   \n",
       "min        0.000000                 1.000000                    0.000000  ...   \n",
       "25%        0.000000                 1.000000                    0.000000  ...   \n",
       "50%        1.000000                 2.000000                    1.000000  ...   \n",
       "75%        1.000000                 3.000000                    1.000000  ...   \n",
       "max        1.000000                 4.000000                    2.000000  ...   \n",
       "\n",
       "       stage_event_system_version  stage_event_tnm_categories  \\\n",
       "count                 3650.000000                 3650.000000   \n",
       "mean                     1.219452                   12.667671   \n",
       "std                      0.674725                   12.582748   \n",
       "min                      0.000000                    0.000000   \n",
       "25%                      1.000000                    4.000000   \n",
       "50%                      1.000000                    8.000000   \n",
       "75%                      2.000000                   22.000000   \n",
       "max                      2.000000                   48.000000   \n",
       "\n",
       "       primary_pathology_esophageal_tumor_cental_location  \\\n",
       "count                                        3650.000000    \n",
       "mean                                            0.355616    \n",
       "std                                             0.572602    \n",
       "min                                             0.000000    \n",
       "25%                                             0.000000    \n",
       "50%                                             0.000000    \n",
       "75%                                             1.000000    \n",
       "max                                             3.000000    \n",
       "\n",
       "       primary_pathology_esophageal_tumor_involvement_sites  \\\n",
       "count                                        3650.000000      \n",
       "mean                                            0.476164      \n",
       "std                                             0.849036      \n",
       "min                                             0.000000      \n",
       "25%                                             0.000000      \n",
       "50%                                             0.000000      \n",
       "75%                                             1.000000      \n",
       "max                                             5.000000      \n",
       "\n",
       "       primary_pathology_histological_type  \\\n",
       "count                          3650.000000   \n",
       "mean                              0.530685   \n",
       "std                               0.499126   \n",
       "min                               0.000000   \n",
       "25%                               0.000000   \n",
       "50%                               1.000000   \n",
       "75%                               1.000000   \n",
       "max                               1.000000   \n",
       "\n",
       "       primary_pathology_neoplasm_histologic_grade  \\\n",
       "count                                  3650.000000   \n",
       "mean                                      1.158082   \n",
       "std                                       0.924942   \n",
       "min                                       0.000000   \n",
       "25%                                       0.000000   \n",
       "50%                                       1.000000   \n",
       "75%                                       2.000000   \n",
       "max                                       3.000000   \n",
       "\n",
       "       primary_pathology_age_at_initial_pathologic_diagnosis  \\\n",
       "count                                        3650.000000       \n",
       "mean                                           62.428493       \n",
       "std                                            11.937696       \n",
       "min                                            27.000000       \n",
       "25%                                            54.000000       \n",
       "50%                                            60.000000       \n",
       "75%                                            72.000000       \n",
       "max                                            90.000000       \n",
       "\n",
       "       primary_pathology_year_of_initial_pathologic_diagnosis  \\\n",
       "count                                        3650.000000        \n",
       "mean                                            6.567945        \n",
       "std                                             3.972903        \n",
       "min                                             0.000000        \n",
       "25%                                             4.000000        \n",
       "50%                                             6.000000        \n",
       "75%                                            10.000000        \n",
       "max                                            15.000000        \n",
       "\n",
       "       primary_pathology_initial_pathologic_diagnosis_method  \\\n",
       "count                                        3650.000000       \n",
       "mean                                            1.021370       \n",
       "std                                             0.709396       \n",
       "min                                             0.000000       \n",
       "25%                                             1.000000       \n",
       "50%                                             1.000000       \n",
       "75%                                             1.000000       \n",
       "max                                             3.000000       \n",
       "\n",
       "       primary_pathology_primary_lymph_node_presentation_assessment  \n",
       "count                                        3650.000000             \n",
       "mean                                            0.443836             \n",
       "std                                             0.780164             \n",
       "min                                             0.000000             \n",
       "25%                                             0.000000             \n",
       "50%                                             0.000000             \n",
       "75%                                             1.000000             \n",
       "max                                             2.000000             \n",
       "\n",
       "[8 rows x 27 columns]"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "1b034913",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:16.876995Z",
     "iopub.status.busy": "2024-10-15T08:18:16.876541Z",
     "iopub.status.idle": "2024-10-15T08:18:16.895388Z",
     "shell.execute_reply": "2024-10-15T08:18:16.894237Z"
    },
    "papermill": {
     "duration": 0.039443,
     "end_time": "2024-10-15T08:18:16.899088",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.859645",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "days_to_birth\n",
      "max: -10143\n",
      "min: -32972\n",
      "width: 22829\n",
      "variance: 18900737.707807768\n",
      "median: -22163.5\n",
      "***\n",
      "gender\n",
      "max: 1\n",
      "min: 0\n",
      "width: 1\n",
      "variance: 0.12567130793769937\n",
      "median: 0.0\n",
      "***\n",
      "height\n",
      "max: 202.0\n",
      "min: 145.0\n",
      "width: 57.0\n",
      "variance: 78.1249468932763\n",
      "median: 172.14555873925502\n",
      "***\n",
      "weight\n",
      "max: 138.0\n",
      "min: 41.0\n",
      "width: 97.0\n",
      "variance: 299.7538609758859\n",
      "median: 72.0\n",
      "***\n",
      "country_of_procurement\n",
      "max: 9\n",
      "min: 0\n",
      "width: 9\n",
      "variance: 6.133878776505911\n",
      "median: 2.0\n",
      "***\n",
      "other_dx\n",
      "max: 1\n",
      "min: 0\n",
      "width: 1\n",
      "variance: 0.09543253893788702\n",
      "median: 0.0\n",
      "***\n",
      "person_neoplasm_cancer_status\n",
      "max: 1\n",
      "min: 0\n",
      "width: 1\n",
      "variance: 0.23738224807656216\n",
      "median: 1.0\n",
      "***\n",
      "vital_status\n",
      "max: 1\n",
      "min: 0\n",
      "width: 1\n",
      "variance: 0.22039714768249205\n",
      "median: 1.0\n",
      "***\n",
      "tobacco_smoking_history\n",
      "max: 4.0\n",
      "min: 1.0\n",
      "width: 3.0\n",
      "variance: 1.2041238289544904\n",
      "median: 2.0\n",
      "***\n",
      "alcohol_history_documented\n",
      "max: 2\n",
      "min: 0\n",
      "width: 2\n",
      "variance: 0.2291543629198724\n",
      "median: 1.0\n",
      "***\n",
      "has_new_tumor_events_information\n",
      "max: 1\n",
      "min: 0\n",
      "width: 1\n",
      "variance: 0.24075241133420902\n",
      "median: 1.0\n",
      "***\n",
      "day_of_form_completion\n",
      "max: 30\n",
      "min: 1\n",
      "width: 29\n",
      "variance: 65.9284765622068\n",
      "median: 16.0\n",
      "***\n",
      "month_of_form_completion\n",
      "max: 12\n",
      "min: 1\n",
      "width: 11\n",
      "variance: 14.00861129667855\n",
      "median: 3.0\n",
      "***\n",
      "year_of_form_completion\n",
      "max: 2015\n",
      "min: 2012\n",
      "width: 3\n",
      "variance: 0.356278851566898\n",
      "median: 2014.0\n",
      "***\n",
      "has_follow_ups_information\n",
      "max: 1\n",
      "min: 0\n",
      "width: 1\n",
      "variance: 0.19107742540814415\n",
      "median: 1.0\n",
      "***\n",
      "has_drugs_information\n",
      "max: 1\n",
      "min: 0\n",
      "width: 1\n",
      "variance: 0.17702600863201354\n",
      "median: 0.0\n",
      "***\n",
      "has_radiations_information\n",
      "max: 1\n",
      "min: 0\n",
      "width: 1\n",
      "variance: 0.19878768999812343\n",
      "median: 0.0\n",
      "***\n",
      "stage_event_system_version\n",
      "max: 2\n",
      "min: 0\n",
      "width: 2\n",
      "variance: 0.45512846687933944\n",
      "median: 1.0\n",
      "***\n",
      "stage_event_tnm_categories\n",
      "max: 48\n",
      "min: 0\n",
      "width: 48\n",
      "variance: 158.2821603302683\n",
      "median: 8.0\n",
      "***\n",
      "primary_pathology_esophageal_tumor_cental_location\n",
      "max: 3\n",
      "min: 0\n",
      "width: 3\n",
      "variance: 0.3277835241133421\n",
      "median: 0.0\n",
      "***\n",
      "primary_pathology_esophageal_tumor_involvement_sites\n",
      "max: 5\n",
      "min: 0\n",
      "width: 5\n",
      "variance: 0.7206647401013325\n",
      "median: 0.0\n",
      "***\n",
      "primary_pathology_histological_type\n",
      "max: 1\n",
      "min: 0\n",
      "width: 1\n",
      "variance: 0.2490584349784199\n",
      "median: 1.0\n",
      "***\n",
      "primary_pathology_neoplasm_histologic_grade\n",
      "max: 3\n",
      "min: 0\n",
      "width: 3\n",
      "variance: 0.8552839932445112\n",
      "median: 1.0\n",
      "***\n",
      "primary_pathology_age_at_initial_pathologic_diagnosis\n",
      "max: 90\n",
      "min: 27\n",
      "width: 63\n",
      "variance: 142.4695443047476\n",
      "median: 60.0\n",
      "***\n",
      "primary_pathology_year_of_initial_pathologic_diagnosis\n",
      "max: 15\n",
      "min: 0\n",
      "width: 15\n",
      "variance: 15.779630024394823\n",
      "median: 6.0\n",
      "***\n",
      "primary_pathology_initial_pathologic_diagnosis_method\n",
      "max: 3\n",
      "min: 0\n",
      "width: 3\n",
      "variance: 0.5031049727903923\n",
      "median: 1.0\n",
      "***\n",
      "primary_pathology_primary_lymph_node_presentation_assessment\n",
      "max: 2\n",
      "min: 0\n",
      "width: 2\n",
      "variance: 0.6084893976355789\n",
      "median: 0.0\n",
      "***\n"
     ]
    }
   ],
   "source": [
    "from wolta.data_tools import stat_sum\n",
    "\n",
    "stat_sum(df,\n",
    "        ['max', 'min', 'width', 'var', 'med'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "6d81acc8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:16.933556Z",
     "iopub.status.busy": "2024-10-15T08:18:16.933135Z",
     "iopub.status.idle": "2024-10-15T08:18:17.130193Z",
     "shell.execute_reply": "2024-10-15T08:18:17.129036Z"
    },
    "papermill": {
     "duration": 0.218917,
     "end_time": "2024-10-15T08:18:17.134051",
     "exception": false,
     "start_time": "2024-10-15T08:18:16.915134",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: ylabel='count'>"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df['person_neoplasm_cancer_status'].value_counts().plot(kind='pie')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "15afbafc",
   "metadata": {
    "papermill": {
     "duration": 0.025749,
     "end_time": "2024-10-15T08:18:17.195069",
     "exception": false,
     "start_time": "2024-10-15T08:18:17.169320",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Data Preparation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "af2ecbca",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:17.230024Z",
     "iopub.status.busy": "2024-10-15T08:18:17.229104Z",
     "iopub.status.idle": "2024-10-15T08:18:17.236700Z",
     "shell.execute_reply": "2024-10-15T08:18:17.235536Z"
    },
    "papermill": {
     "duration": 0.027703,
     "end_time": "2024-10-15T08:18:17.239308",
     "exception": false,
     "start_time": "2024-10-15T08:18:17.211605",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "y = df['person_neoplasm_cancer_status'].values\n",
    "del df['person_neoplasm_cancer_status']\n",
    "X = df.values\n",
    "del df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "7e3f085a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:17.273865Z",
     "iopub.status.busy": "2024-10-15T08:18:17.273418Z",
     "iopub.status.idle": "2024-10-15T08:18:18.893775Z",
     "shell.execute_reply": "2024-10-15T08:18:18.892381Z"
    },
    "papermill": {
     "duration": 1.641161,
     "end_time": "2024-10-15T08:18:18.897045",
     "exception": false,
     "start_time": "2024-10-15T08:18:17.255884",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "del X, y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "1ce2d97b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:18.933279Z",
     "iopub.status.busy": "2024-10-15T08:18:18.932463Z",
     "iopub.status.idle": "2024-10-15T08:18:18.941346Z",
     "shell.execute_reply": "2024-10-15T08:18:18.940129Z"
    },
    "papermill": {
     "duration": 0.030121,
     "end_time": "2024-10-15T08:18:18.944057",
     "exception": false,
     "start_time": "2024-10-15T08:18:18.913936",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Counter({1: 1788, 0: 1132})\n",
      "Counter({1: 447, 0: 283})\n"
     ]
    }
   ],
   "source": [
    "from collections import Counter\n",
    "\n",
    "print(Counter(y_train))\n",
    "print(Counter(y_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "55759ab1",
   "metadata": {
    "papermill": {
     "duration": 0.017045,
     "end_time": "2024-10-15T08:18:18.978629",
     "exception": false,
     "start_time": "2024-10-15T08:18:18.961584",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "4421399f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:19.015413Z",
     "iopub.status.busy": "2024-10-15T08:18:19.014365Z",
     "iopub.status.idle": "2024-10-15T08:18:24.871129Z",
     "shell.execute_reply": "2024-10-15T08:18:24.869469Z"
    },
    "papermill": {
     "duration": 5.879603,
     "end_time": "2024-10-15T08:18:24.875556",
     "exception": false,
     "start_time": "2024-10-15T08:18:18.995953",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "AdaBoost\n",
      "Accuracy Score: 0.9547945205479452\n",
      "Precision Score: 0.9548223467126972\n",
      "F1 Score (weighted): 0.9546545821986032\n",
      "***\n",
      "CatBoost\n",
      "Accuracy Score: 1.0\n",
      "Precision Score: 1.0\n",
      "F1 Score (weighted): 1.0\n",
      "***\n",
      "LightGBM\n",
      "Accuracy Score: 1.0\n",
      "Precision Score: 1.0\n",
      "F1 Score (weighted): 1.0\n",
      "***\n",
      "Random Forest\n",
      "Accuracy Score: 1.0\n",
      "Precision Score: 1.0\n",
      "F1 Score (weighted): 1.0\n",
      "***\n",
      "Extra Tree\n",
      "Accuracy Score: 0.9945205479452055\n",
      "Precision Score: 0.9945291035496313\n",
      "F1 Score (weighted): 0.9945169480645651\n",
      "***\n",
      "Decision Tree\n",
      "Accuracy Score: 0.9972602739726028\n",
      "Precision Score: 0.9972795001201634\n",
      "F1 Score (weighted): 0.997262025992184\n",
      "***\n",
      "Ridge\n",
      "Accuracy Score: 0.8575342465753425\n",
      "Precision Score: 0.857023853969026\n",
      "F1 Score (weighted): 0.8561176185119007\n",
      "***\n",
      "Perceptron\n",
      "Accuracy Score: 0.5753424657534246\n",
      "Precision Score: 0.58269359743914\n",
      "F1 Score (weighted): 0.5783076373829799\n",
      "***\n"
     ]
    }
   ],
   "source": [
    "from wolta.model_tools import compare_models\n",
    "\n",
    "results = compare_models('clf',\n",
    "                        ['ada', 'cat', 'lbm', 'raf', 'ext', 'dtr', 'rdg', 'per'],\n",
    "                        ['acc', 'precision', 'f1'],\n",
    "                        X_train, y_train, X_test, y_test,\n",
    "                        get_result=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "7d74558a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:24.981266Z",
     "iopub.status.busy": "2024-10-15T08:18:24.974661Z",
     "iopub.status.idle": "2024-10-15T08:18:27.546720Z",
     "shell.execute_reply": "2024-10-15T08:18:27.545372Z"
    },
    "papermill": {
     "duration": 2.621928,
     "end_time": "2024-10-15T08:18:27.549677",
     "exception": false,
     "start_time": "2024-10-15T08:18:24.927749",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best Algorithm is cat with the score of 1.0\n"
     ]
    }
   ],
   "source": [
    "from wolta.model_tools import get_best_model\n",
    "\n",
    "model = get_best_model(results, 'acc', 'clf', X_train, y_train, behavior='max-best')\n",
    "y_pred = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "cdf9fef6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:27.590189Z",
     "iopub.status.busy": "2024-10-15T08:18:27.589636Z",
     "iopub.status.idle": "2024-10-15T08:18:27.609342Z",
     "shell.execute_reply": "2024-10-15T08:18:27.607589Z"
    },
    "papermill": {
     "duration": 0.040778,
     "end_time": "2024-10-15T08:18:27.612134",
     "exception": false,
     "start_time": "2024-10-15T08:18:27.571356",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00       283\n",
      "           1       1.00      1.00      1.00       447\n",
      "\n",
      "    accuracy                           1.00       730\n",
      "   macro avg       1.00      1.00      1.00       730\n",
      "weighted avg       1.00      1.00      1.00       730\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report as rep\n",
    "\n",
    "print(rep(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "58cda0f3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-15T08:18:27.649143Z",
     "iopub.status.busy": "2024-10-15T08:18:27.648297Z",
     "iopub.status.idle": "2024-10-15T08:18:28.041699Z",
     "shell.execute_reply": "2024-10-15T08:18:28.040307Z"
    },
    "papermill": {
     "duration": 0.414711,
     "end_time": "2024-10-15T08:18:28.044377",
     "exception": false,
     "start_time": "2024-10-15T08:18:27.629666",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7f3fedd2dab0>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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PnI7FhInAABgit2wyNJA31VH4gQAAEwxDDefqnOjb23jqToAAAAXMeMEAABMqenN4XUJiRMAADCFxAkAAMBFDXlzOHucAAAAXMSMEwAAMKUhP1VH4gQAAEw5nzi5s8epCoOpYSzVAQAAuIgZJwAAYApP1QEAALjI+G9xp399xVIdAACAi5hxAgAAprBUBwAA4KoGvFZH4gQAAMxxc8ZJ9XjGiT1OAAAALmLGCQAAmMLJ4QAAAC5qyJvDWaoDAABwETNOAADAHMPi3gbvejzjROIEAABMach7nFiqAwAAcBEzTgAAwBwOwAQAAHBNQ36qzqXE6cMPP3R5wNtvv/2ygwEAAKjLXEqc7rjjDpcGs1gsKi0tdSceAABQH9Tj5TZ3uJQ42e326o4DAADUEw15qc6tp+oKCgqqKg4AAFBfGFVQ6inTiVNpaalmzJihq666SgEBATp27Jgk6dlnn9Vbb71V5QECAADUFaYTp+eff16pqamaNWuWfHx8HPUdO3bUm2++WaXBAQCAushSBcV1M2fO1A033KDAwECFhobqjjvu0KFDh5za9OzZUxaLxamMGTPGqU16eroGDhwof39/hYaG6vHHH1dJSYmpWEwnTosXL9bf//53DR06VJ6eno76a6+9Vt98843Z4QAAQH1Tw0t1GzduVHJysrZv3661a9equLhYffr0UX5+vlO70aNHKzMz01FmzZrluFZaWqqBAweqqKhIW7du1dtvv63U1FRNnTrVVCymz3H68ccfFRMTU67ebreruLjY7HAAAACXtGbNGqfPqampCg0N1a5du9SjRw9Hvb+/v2w2W4VjfPrppzpw4IDWrVunsLAwdenSRTNmzNDkyZOVkpLitIp2KaZnnGJjY7V58+Zy9e+++66uu+46s8MBAID6popmnHJzc51KYWGhS7fPycmRJIWEhDjVL1myRM2aNVPHjh01ZcoUnT171nFt27Zt6tSpk8LCwhx1ffv2VW5urvbv3+/yVzc94zR16lQlJSXpxx9/lN1u1/vvv69Dhw5p8eLFWr16tdnhAABAfWNYzhd3+kuKjIx0qn7uueeUkpJyya52u13jxo3TjTfeqI4dOzrq//CHPyg6OloRERHau3evJk+erEOHDun999+XJGVlZTklTZIcn7OyslwO3XTiNGjQIK1atUrTp09X48aNNXXqVP3ud7/TqlWr9Pvf/97scAAAoIHKyMiQ1Wp1fPb19a20T3Jysr7++mtt2bLFqf7BBx90/LlTp04KDw9Xr169dPToUV199dVVFvNlvavu5ptv1tq1a6ssCAAAUH8YxvniTn9JslqtTolTZcaOHavVq1dr06ZNatGixSXbdu3aVZJ05MgRXX311bLZbPr888+d2pw4cUKSLrovqiKX/ZLfnTt36uDBg5LO73uKi4u73KEAAEB94u4hlib7GoahRx55RCtWrFBaWppatWpVaZ89e/ZIksLDwyVJ8fHxev7553Xy5EmFhoZKktauXSur1arY2FiXYzGdOP3www+699579dlnnyk4OFiSlJ2dre7du2vZsmWVZoAAAABmJCcna+nSpfrggw8UGBjo2JMUFBSkRo0a6ejRo1q6dKkGDBigpk2bau/evRo/frx69Oihzp07S5L69Omj2NhYDR8+XLNmzVJWVpaeeeYZJScnu7REWMb0U3UPPPCAiouLdfDgQZ06dUqnTp3SwYMHZbfb9cADD5gdDgAA1Ddlm8PdKSbMnz9fOTk56tmzp8LDwx1l+fLlkiQfHx+tW7dOffr0Ufv27TVx4kQNGTJEq1atcozh6emp1atXy9PTU/Hx8Ro2bJhGjBih6dOnm4rF9IzTxo0btXXrVrVr185R165dO7322mu6+eabzQ4HAADqGYtxvrjT3wyjkg1VkZGR2rhxY6XjREdH66OPPjJ38wuYTpwiIyMrPOiytLRUERERbgUDAADqgRre41SXmF6qe/HFF/XII49o586djrqdO3fqscce00svvVSlwQEAANQlLs04NWnSRBbLr+uR+fn56tq1q7y8zncvKSmRl5eXRo4cqTvuuKNaAgUAAHVEFR2AWR+5lDjNnj27msMAAAD1RgNeqnMpcUpKSqruOAAAAOq8yz4AU5IKCgpUVFTkVGfmBFAAAFAPNeAZJ9Obw/Pz8zV27FiFhoaqcePGatKkiVMBAABXOKMKSj1lOnF64okn9O9//1vz58+Xr6+v3nzzTU2bNk0RERFavHhxdcQIAABQJ5heqlu1apUWL16snj176v7779fNN9+smJgYRUdHa8mSJRo6dGh1xAkAAOqKBvxUnekZp1OnTql169aSzu9nOnXqlCTppptu0qZNm6o2OgAAUOeUnRzuTqmvTCdOrVu31nfffSdJat++vf7xj39IOj8TVfbSXwAAgCuR6cTp/vvv11dffSVJevLJJzV37lz5+flp/Pjxevzxx6s8QAAAUMc04M3hpvc4jR8/3vHn3r1765tvvtGuXbsUExOjzp07V2lwAAAAdYlb5zhJ5980HB0dXRWxAACAesAi9/Yp1d+t4S4mTnPmzHF5wEcfffSygwEAAKjLXEqcXnnlFZcGs1gsJE71XPgfvpGXxbu2wwCqxSfH99R2CEC1yT1jV5O2NXSzBnwcgUuJU9lTdAAAALxyBQAAAJVye3M4AABoYBrwjBOJEwAAMMXd078b1MnhAAAADRUzTgAAwJwGvFR3WTNOmzdv1rBhwxQfH68ff/xRkvS///u/2rJlS5UGBwAA6qAG/MoV04nTe++9p759+6pRo0bavXu3CgsLJUk5OTl64YUXqjxAAACAusJ04vSnP/1JCxYs0BtvvCFv718PSrzxxhv15ZdfVmlwAACg7inbHO5Oqa9M73E6dOiQevToUa4+KChI2dnZVRETAACoyxrwyeGmZ5xsNpuOHDlSrn7Lli1q3bp1lQQFAADqMPY4uW706NF67LHHtGPHDlksFh0/flxLlizRpEmT9PDDD1dHjAAAAHWC6aW6J598Una7Xb169dLZs2fVo0cP+fr6atKkSXrkkUeqI0YAAFCHNOQDME0nThaLRU8//bQef/xxHTlyRHl5eYqNjVVAQEB1xAcAAOqaBnyO02UfgOnj46PY2NiqjAUAAKBOM73HKTExUbfccstFCwAAuMK5exSByRmnmTNn6oYbblBgYKBCQ0N1xx136NChQ05tCgoKlJycrKZNmyogIEBDhgzRiRMnnNqkp6dr4MCB8vf3V2hoqB5//HGVlJSYisV04tSlSxdde+21jhIbG6uioiJ9+eWX6tSpk9nhAABAfVPDT9Vt3LhRycnJ2r59u9auXavi4mL16dNH+fn5jjbjx4/XqlWr9M9//lMbN27U8ePHNXjwYMf10tJSDRw4UEVFRdq6davefvttpaamaurUqaZiMb1U98orr1RYn5KSory8PLPDAQCABio3N9fps6+vr3x9fcu1W7NmjdPn1NRUhYaGateuXerRo4dycnL01ltvaenSpY7Vr0WLFqlDhw7avn27unXrpk8//VQHDhzQunXrFBYWpi5dumjGjBmaPHmyUlJS5OPj41LMl/WuuooMGzZMCxcurKrhAABAXVVFM06RkZEKCgpylJkzZ7p0+5ycHElSSEiIJGnXrl0qLi5W7969HW3at2+vqKgobdu2TZK0bds2derUSWFhYY42ffv2VW5urvbv3+/yV7/szeEX2rZtm/z8/KpqOAAAUEdV1XEEGRkZslqtjvqKZpsuZLfbNW7cON14443q2LGjJCkrK0s+Pj4KDg52ahsWFqasrCxHm98mTWXXy665ynTi9Nv1QkkyDEOZmZnauXOnnn32WbPDAQCABspqtTolTq5ITk7W119/rS1btlRTVJdmOnEKCgpy+uzh4aF27dpp+vTp6tOnT5UFBgAA8Ftjx47V6tWrtWnTJrVo0cJRb7PZVFRUpOzsbKdZpxMnTshmsznafP75507jlT11V9bGFaYSp9LSUt1///3q1KmTmjRpYqYrAAC4UtTwAZiGYeiRRx7RihUrlJaWplatWjldj4uLk7e3t9avX68hQ4ZIkg4dOqT09HTFx8dLkuLj4/X888/r5MmTCg0NlSStXbtWVqvV1LmUphInT09P9enTRwcPHiRxAgCggarpV64kJydr6dKl+uCDDxQYGOjYkxQUFKRGjRopKChIo0aN0oQJExQSEiKr1apHHnlE8fHx6tatmySpT58+io2N1fDhwzVr1ixlZWXpmWeeUXJyskt7q8qYfqquY8eOOnbsmNluAAAAl2X+/PnKyclRz549FR4e7ijLly93tHnllVd06623asiQIerRo4dsNpvef/99x3VPT0+tXr1anp6eio+P17BhwzRixAhNnz7dVCym9zj96U9/0qRJkzRjxgzFxcWpcePGTtfNbvICAAD1UA2+b84wKr+Zn5+f5s6dq7lz5160TXR0tD766CO3YnE5cZo+fbomTpyoAQMGSJJuv/12WSwWx3XDMGSxWFRaWupWQAAAoI7jJb+VmzZtmsaMGaMNGzZUZzwAAAB1lsuJU9k0WUJCQrUFAwAA6r6a3hxel5ja4/TbpTkAANBAsVTnmrZt21aaPJ06dcqtgAAAAOoqU4nTtGnTyp0cDgAAGhaW6lx0zz33OE7bBAAADVQDXqpz+QBM9jcBAICGzvRTdQAAoIFrwDNOLidOdru9OuMAAAD1BHucAAAAXNWAZ5xMv+QXAACgoWLGCQAAmNOAZ5xInAAAgCkNeY8TS3UAAAAuYsYJAACYw1IdAACAa1iqAwAAQKWYcQIAAOawVAcAAOCiBpw4sVQHAADgImacAACAKZb/Fnf611ckTgAAwJwGvFRH4gQAAEzhOAIAAABUihknAABgDkt1AAAAJtTj5McdLNUBAAC4iBknAABgSkPeHE7iBAAAzGnAe5xYqgMAAHXepk2bdNtttykiIkIWi0UrV650un7ffffJYrE4lX79+jm1OXXqlIYOHSqr1arg4GCNGjVKeXl5puIgcQIAAKaULdW5U8zKz8/Xtddeq7lz5160Tb9+/ZSZmeko77zzjtP1oUOHav/+/Vq7dq1Wr16tTZs26cEHHzQVB0t1AADAnFpYquvfv7/69+9/yTa+vr6y2WwVXjt48KDWrFmjL774Qtdff70k6bXXXtOAAQP00ksvKSIiwqU4mHECAAC1Ijc316kUFha6NV5aWppCQ0PVrl07Pfzww/rll18c17Zt26bg4GBH0iRJvXv3loeHh3bs2OHyPUicAACAKVW1VBcZGamgoCBHmTlz5mXH1K9fPy1evFjr16/XX/7yF23cuFH9+/dXaWmpJCkrK0uhoaFOfby8vBQSEqKsrCyX78NSHQAAMKeKluoyMjJktVod1b6+vpc95D333OP4c6dOndS5c2ddffXVSktLU69evS573Asx4wQAAMwxqqBIslqtTsWdxOlCrVu3VrNmzXTkyBFJks1m08mTJ53alJSU6NSpUxfdF1UREicAAHDF+eGHH/TLL78oPDxckhQfH6/s7Gzt2rXL0ebf//637Ha7unbt6vK4LNUBAABTauPk8Ly8PMfskSR999132rNnj0JCQhQSEqJp06ZpyJAhstlsOnr0qJ544gnFxMSob9++kqQOHTqoX79+Gj16tBYsWKDi4mKNHTtW99xzj8tP1EnMOAEAALOqaKnOjJ07d+q6667TddddJ0maMGGCrrvuOk2dOlWenp7au3evbr/9drVt21ajRo1SXFycNm/e7LT8t2TJErVv3169evXSgAEDdNNNN+nvf/+7qTiYcQIAAHVez549ZRgXz7g++eSTSscICQnR0qVL3YqDxAkAAJhiMQxZLpHEuNK/viJxAgAA5vCSXwAAAFSGGScAAGBKbTxVV1eQOAEAAHNYqgMAAEBlmHECAACmsFQHAADgqga8VEfiBAAATGnIM07scQIAAHARM04AAMAcluoAAABcV5+X29zBUh0AAICLmHECAADmGMb5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+HB5e3vrgw8+0FNPPaWJEyeqa9euF43LYrHo0UcfVUpKih566CH5+/tXyfe92LglJSXKysoq1/a3S3Fnz54t18bX11dNmjSpkthQdT5c1EwfLmpWeUOgjrtw39KFPjm+p8L62au+rYZoUFMa8lJdnZ9xkqRXX31VSUlJmjx5sq655hrdd9996ty5s1atWiVLRUfT6nxS0axZs0s+nTZu3DitWLFCmzdv1vXXX6+OHTtq6dKlmj9/vl566aVK40pKSlJxcbH+9re/XfZ3c3Xc/fv3Kzw83KlER0c79XvjjTfKtbnUrBkAAJelFp6q27Rpk2677TZFRETIYrFo5cqVziEZhqZOnarw8HA1atRIvXv31rffOifop06d0tChQ2W1WhUcHKxRo0YpL6/8eWSXYjGMerzQiCqTm5uroKAg9dQgeVlYMsKV6WKzH8CVIPeMXU3ant/GcrHVFrfv8d/fFfH9psvL2++yxykpLtC2NVNNxfrxxx/rs88+U1xcnAYPHqwVK1bojjvucFz/y1/+opkzZ+rtt99Wq1at9Oyzz2rfvn06cOCA/PzOx9q/f39lZmbq9ddfV3Fxse6//37dcMMNWrp0qcux1/mlOgAAULfUxlJd//791b9//wqvGYah2bNn65lnntGgQYMkSYsXL1ZYWJhWrlype+65RwcPHtSaNWv0xRdf6Prrr5d0/g0iAwYM0EsvvVTuQbGLqRdLdQAAoA6xG+4XqdwbLAoLCy8rnO+++05ZWVnq3bu3oy4oKEhdu3bVtm3bJEnbtm1TcHCwI2mSpN69e8vDw8Olt4CUIXECAADmVNEep8jISKe3WMycOfOywil7MOrCswvDwsIc17KyshQaGup03cvLSyEhIeUerLoUluoAAECtyMjIcNrjVB8OZmbGCQAAmGKRmy/5/e84VqvVqVxu4mSz2SRJJ06ccKo/ceKE45rNZtPJkyedrpeUlOjUqVOONq4gcQIAAOaUnRzuTqlCrVq1ks1m0/r16x11ubm52rFjh+Lj4yVJ8fHxys7OdjpQ+9///rfsdvslz228EEt1AACgzsvLy9ORI78euPrdd99pz549CgkJUVRUlMaNG6c//elPatOmjeM4goiICMeRBR06dFC/fv00evRoLViwQMXFxRo7dqzuuecel5+ok0icAACASbVxHMHOnTud3iE7YcIESecPjU5N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WrlypctjpqSkqEuXLm7F9f3338tisWjPnj1ujQPg8pE4Abik++67TxaLRRaLxfHW+enTp6ukpKTa7/3+++9rxowZLrV1JdkBAHfxkl8AlerXr58WLVqkwsJCffTRR0pOTpa3t7emTJlSrm1RUdEl33lmRkhISJWMAwBVhRknAJXy9fWVzWZTdHS0Hn74YfXu3VsffvihpF+X155//nlFRESoXbt2kqSMjAzdfffdCg4OVkhIiAYNGqTvv//eMWZpaakmTJig4OBgNW3aVE888YQufHXmhUt1hYWFmjx5siIjI+Xr66uYmBi99dZb+v7775WYmChJatKkiSwWi+677z5Jkt1u18yZM9WqVSs1atRI1157rd59912n+3z00Udq27atGjVqpMTERKc4XTV58mS1bdtW/v7+at26tZ599lkVFxeXa/f6668rMjJS/v7+uvvuu5WTk+N0/c0331SHDh3k5+en9u3ba968eaZjAVB9SJwAmNaoUSMVFRU5Pq9fv16HDh3S2rVrtXr1ahUXF6tv374KDAzU5s2b9dlnnykgIED9+vVz9Hv55ZeVmpqqhQsXasuWLTp16pRWrFhxyfuOGDFC77zzjubMmaODBw/q9ddfV0BAgCIjI/Xee+9Jkg4dOqTMzEy9+uqrkqSZM2dq8eLFWrBggfbv36/x48dr2LBh2rhxo6TzCd7gwYN12223ac+ePXrggQf05JNPmv7fJDAwUKmpqTpw4IBeffVVvfHGG3rllVec2hw5ckT/+Mc/tGrVKq1Zs0a7d+/WH//4R8f1JUuWaOrUqXr++ed18OBBvfDCC3r22Wf19ttvm44HQDUxAOASkpKSjEGDBhmGYRh2u91Yu3at4evra0yaNMlxPSwszCgsLHT0+d///V+jXbt2ht1ud9QVFhYajRo1Mj755BPDMAwjPDzcmDVrluN6cXGx0aJFC8e9DMMwEhISjMcee8wwDMM4dOiQIclYu3ZthXFu2LDBkGScPn3aUVdQUGD4+/sbW7dudWo7atQo49577zUMwzCmTJlixMbGOl2fPHlyubEuJMlYsWLFRa+/+OKLRlxcnOPzc889Z3h6eho//PCDo+7jjz82PDw8jMzMTMMwDOPqq682li5d6jTOjBkzjPj4eMMwDOO7774zJBm7d+++6H0BVC/2OAGo1OrVqxUQEKDi4mLZ7Xb94Q9/UEpKiuN6p06dnPY1ffXVVzpy5IgCAwOdxikoKNDRo0eVk5OjzMxMde3a1XHNy8tL119/fbnlujJ79uyRp6enEhISXI77yJEjOnv2rH7/+9871RcVFem6666TJB08eNApDkmKj493+R5lli9frjlz5ujo0aPKy8tTSUmJrFarU5uoqChdddVVTvex2+06dOiQAgMDdfToUY0aNUqjR492tCkpKVFQUJDpeABUDxInAJVKTEzU/Pnz5ePjo4iICHl5Of/T0bhxY6fPeXl5iouL05IlS8qN1bx588uKoVGjRqb75OXlSZL+9a9/OSUs0vl9W1Vl27ZtGjp0qKZNm6a+ffsqKChIy5Yt08svv2w61jfeeKNcIufp6VllsQJwD4kTgEo1btxYMTExLrf/3e9+p+XLlys0NLTcrEuZ8PBw7dixQz169JB0fmZl165d+t3vfldh+06dOslut2vjxo3q3bt3uetlM16lpaWOutjYWPn6+io9Pf2iM1UdOnRwbHQvs3379sq/5G9s3bpV0dHRevrppx11//nPf8q1S09P1/HjxxUREeG4j4eHh9q1a6ewsDBFRETo2LFjGjp0qKn7A6g5bA4HUOWGDh2qZs2aadCgQdq8ebO+++47paWl6dFHH9UPP/wgSXrsscf05z//WStXrtQ333yjP/7xj5c8g6lly5ZKSkrSyJEjtXLlSseY//jHPyRJ0dHRslgsWr16tX766Sfl5eUpMDBQkyZN0vjx4/X222/r6NGj+vLLL/Xaa685NlyPGTNG3377rR5//HEdOnRIS5cuVWpqqqnv26ZNG6Wnp2vZsmU6evSo5syZU+FGdz8/PyUlJemrr77S5s2b9eijj+ruu++WzWaTJE2bNk0zZ87UnDlzdPjwYe3bt0+LFi3SX//6V1PxAKg+JE4Aqpy/v782bdqkqKgoDR48WB06dNCoUaNUUFDgmIGaOHGihg8frqSkJMXHxyswMFB33nnnJcedP3++7rrrLv3xj39U+/btNXr0aOXn50uSrrrqKk2bNk1PPvmkwsLCNHbsWEnSjBkz9Oyzz2rmzJnq0KGD+vXrp3/9619q1aqVpPP7jt577z2tXLlS1157rRYsWKAXXnjB1Pe9/fbbNX78eI0dO1ZdunTR1q1b9eyzz5ZrFxMTo8GDB2vAgAHq06ePOnfu7HTcwAMPPKA333xTixYtUqdOnZSQkKDU1FRHrABqn8W42E5MAAAAOGHGCQAAwEUkTgAAAC4icQIAAHARiRMAAICLSJwAAABcROIEAADgIhInAAAAF5E4AQAAuIjECQAAwEUkTgAAAC4icQIAAHDR/w88wnomnw7tkAAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix as conf\n",
    "from sklearn.metrics import ConfusionMatrixDisplay as cmd\n",
    "\n",
    "cm = conf(y_test, y_pred)\n",
    "disp = cmd(confusion_matrix=cm, display_labels=outs)\n",
    "disp.plot()"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "datasetId": 5875096,
     "sourceId": 9624910,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 30786,
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 33.266238,
   "end_time": "2024-10-15T08:18:28.884631",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2024-10-15T08:17:55.618393",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
