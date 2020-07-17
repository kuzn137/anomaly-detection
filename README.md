# Anomaly-detection
Resource Utilization Anomaly Detection
Developed CPU utilization data (CUD) algorithm and program that detect potential anomalies in real-time.
## Libraries
os, sys, statsmodels, pandas, scikitlearn
## Autor
Inga Kuznetsova
## Files
anomaly_detection.py, anomaly_detection_part2.py, IngaKuznetsovaProjectOverview.doc, IngaKuznetsovaProjectOverviewpart2.doc
## Summary
Data are from privetly shared file data.csv.

My program in general works as follows: It waits first 400 time intervals before it produces first results. After that program returns the list of last M outliers ordered by time stamp. 
