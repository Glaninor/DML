library(dplyr)
library(tidyverse)
library(data.table)
library(keras)
library(tensorflow)


apache <- data.table(read.csv("apachePatientResult.csv"))
patient <- data.table(read.csv("patient.csv"))
admissiondx <- data.table(read.csv("admissionDx.csv"))
diagnosis <- data.table(read.csv("diagnosis.csv"))
infusionDrug <- data.table(read.csv("infusionDrug.csv"))
lab <- data.table(read.csv("lab.csv"))
medication <- data.table(read.csv("medication.csv"))
treatment <- data.table(read.csv("treatment.csv"))


patientData <- left_join(patient, apache, by="patientunitstayid")
levels(patientData$actualicumortality) <- c(1,0)
levels(patientData$actualhospitalmortality) <- c(1,0)


diagnosis <- diagnosis[-which(diagnosis$diagnosisoffset>=2880),]
Diagnosis <- Diagnosis[1:round(nrow(Diagnosis)/10),] 
infusionDrug <- infusionDrug[-which(infusionDrug$infusionoffset>=2880),]
InfusionDrug <- InfusionDrug[1:round(nrow(InfusionDrug)/10),] 
lab <- lab[-which(lab$labresultoffset>=2880),]
lab <- lab[-which(lab$labresultrevisedoffset>=2880),]
Lab <- lab[1:round(nrow(Lab)/10),] 
medication <- medication[-which(medication$drugstartoffset>=2880),]
medication <- medication[-which(medication$drugstopoffset>=2880),]
medication <- medication[-which(medication$drugorderoffset>=2880),]
Medication <- medication[1:round(nrow(Medication)/10),] 
treatment <- treatment[-which(treatment$treatmentoffset>=2880),]
Treatment <- treatment[1:round(nrow(Treatment)/10),] 


p.diagnosis<-left_join(patientData,diagnosis)
gc()
neicu <- nrow(as.data.frame(table(p.diagnosis$patientunitstayid)))
neicu
p.lab<-left_join(patientData,lab)
gc()
neicu <- nrow(as.data.frame(table(p.lab$patientunitstayid)))
neicu
p.medication<-left_join(patientData,medication)
gc()
neicu <- nrow(as.data.frame(table(p.medication$patientunitstayid)))
neicu
p.treatment<-left_join(patientData,treatment)
gc()
neicu <- nrow(as.data.frame(table(p.treatment$patientunitstayid)))
neicu
p.infusionDrug<-left_join(patientData,infusionDrug)
gc()
neicu <- nrow(as.data.frame(table(p.infusionDrug$patientunitstayid)))
neicu

nDiagnosis <- nrow(as.data.frame(table(Diagnosis$patientunitstayid)))
nInfusionDrug <- nrow(as.data.frame(table(InfusionDrug$patientunitstayid)))
nMedication <- nrow(as.data.frame(table(Medication$patientunitstayid)))
nLab <- nrow(as.data.frame(table(Lab$patientunitstayid)))
nTreatment <- nrow(as.data.frame(table(Treatment$patientunitstayid)))
napache <- nrow(as.data.frame(table(apache$patientunitstayid)))
nP <- as.data.frame(rbind(nDiagnosis,nInfusionDrug,nLab,nMedication,nTreatment,napache))


ndiagnosis <- nrow(as.data.frame(table(diagnosis$patientunitstayid)))
ninfusionDrug <- nrow(as.data.frame(table(infusionDrug$patientunitstayid)))
nmedication <- nrow(as.data.frame(table(medication$patientunitstayid)))
nlab <- nrow(as.data.frame(table(lab$patientunitstayid)))
ntreatment <- nrow(as.data.frame(table(treatment$patientunitstayid)))
napache <- nrow(as.data.frame(table(apache$patientunitstayid)))
np <- as.data.frame(rbind(ndiagnosis,ninfusionDrug,nlab,nmedication,ntreatment,napache))
NP <- cbind(nP,np)
NP
rm(Diagnosis,Lab,Medication,Treatment,patient,apache)


p.diagnosis <- na.omit(p.diagnosis)
ntrain.diagnosis = sample(nrow(p.diagnosis),floor(0.7*nrow(p.diagnosis)),replace=FALSE)
train.diagnosis = p.diagnosis[ntrain.diagnosis,]
test.diagnosis = p.diagnosis[-ntrain.diagnosis,]
ntrain.infusionDrug = sample(nrow(p.infusionDrug),floor(0.7*nrow(p.infusionDrug)),replace=FALSE)
train.infusionDrug = p.infusionDrug[ntrain.infusionDrug,]
test.infusionDrug = p.infusionDrug[-ntrain.infusionDrug,]
ntrain.lab = sample(nrow(p.lab),floor(0.7*nrow(p.lab)),replace=FALSE)
train.lab = p.lab[ntrain.lab,]
test.lab = p.lab[-ntrain.lab,]
ntrain.medication = sample(nrow(p.medication),floor(0.7*nrow(p.medication)),replace=FALSE)
train.medication = p.medication[ntrain.medication,]
test.medication = p.medication[-ntrain.medication,]
ntrain.treatment = sample(nrow(p.treatment),floor(0.7*nrow(p.treatment)),replace=FALSE)
train.treatment = p.treatment[ntrain.treatment,]
test.treatment = p.treatment[-ntrain.treatment,]


keras.train.d <- as.matrix(train.diagnosis[,c(2:36,38:57)])
keras.train.d.l <- as.matrix(train.diagnosis[,37])
keras.test.d <- as.matrix(test.diagnosis[,c(2:36,38:57)])
keras.test.d.l <- as.matrix(test.diagnosis[,37])


use_condaenv("r-tensorflow")
model <- keras_model_sequential()

model %>%
layer_dense(units = 64, activation = 'relu', input_shape = c(56)) %>%
layer_dropout(rate = 0.1) %>%
layer_dense(units = 32, activation = 'relu') %>%
layer_dropout(rate = 0.1) %>%
layer_dense(units = 1, activation = 'sigmoid') %>%
compile(
optimizer = optimizer_rmsprop(),
loss = loss_binary_crossentropy,
metrics = c('accuracy')
)

history <- model %>% fit(keras.train.d,keras.train.d.l, epochs=30, batch_size=64)
keras.pre <- predict(model,newdata = keras.test.d)
model %>% evaluate(keras.test,keras.test.d.l)
plot(history)

