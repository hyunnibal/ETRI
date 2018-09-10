dir<-dir()
csv = list()
for(i in 1:length(dir)){
  csv[[i]] = read.csv(dir[i],sep = ',', header = TRUE)
  print(dim(csv[[i]]))
}
csv[[1]]<-csv[[1]][1:5203,]
data_total = NULL
for(i in 1:13){
  data_total = rbind(data_total, csv[[i]])
  print(dim(data_total))
}
colnames(csv[[14]])<-colnames(csv[[15]][-1])
label_total = NULL
for(i in 1:13){
  if(i==1)
    label_total = rbind(label_total, csv[[13+i]])
  else
    label_total = rbind(label_total, csv[[13+i]][,-1])
  print(dim(label_total))
}

write.csv(data_total[-c(1,2),],file='eicu_data_0910.csv')
write.csv(label_total,file='eicu_label_0910.csv')