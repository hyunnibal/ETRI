data_total<-NULL
label_total<-NULL
for(k in 5001:17000){
  data<-read.csv('../data/vitalPeriodic.csv',nrow=10000,stringsAsFactors = FALSE, na.strings = 'NA',skip=1+9000*k)
  patient<-unique(data[,1])

  for(i in 1:length(patient)){
    data_p = data[grep(patient[i],data[,1]),]
    sorted_data_p = data_p[order(data_p[,6]),]
    #print(sorted_data_p[1,6])
  
    if(sorted_data_p[1,6]<15&&is.na(sorted_data_p[1:60,15])<45){
      if(nrow(sorted_data_p)>60 && sorted_data_p[60,6]==sorted_data_p[1,6]+295){
        print(i)
        data_c = sorted_data_p[1:60,c(1,6,8,9,10,15)]
        x=data_c[1,1]
        for(j in 1:20){
          temp<-data_c[(j*3-2):(j*3),3:6]
          x<-cbind(x,median(temp[,1],na.rm=T),median(temp[,2],na.rm=T),median(temp[,3],na.rm=T),median(temp[,4],na.rm=T))
        }
        if(sum(is.na(x)) == 0){
          label=cbind(as.logical(sum(x[seq(66,81,4)]<95)),as.logical(sum(x[seq(67,81,4)]<70)),as.logical(sum(x[seq(67,81,4)]>100)),as.logical(sum(x[seq(68,81,4)]<13)),as.logical(sum(x[seq(68,81,4)]>20)),as.logical(sum(x[seq(69,81,4)]<70)),as.logical(sum(x[seq(69,81,4)]>110)))
          data_total<-rbind(data_total,x[1:65])
          label_total<-rbind(label_total,label)
          message('inserted = ',nrow(data_total))
        }
      }
    }
  }
  message('k = ',k)
  rm(list='data')
  rm(list='data_p')
  rm(list='data_c')
  rm(list='sorted_data_p')
  if(k%%1000==0){
    write.csv(data_total,file=sprintf("../data/data_total%d.csv",k))
    write.csv(label_total,file=sprintf("../data/label_total%d.csv",k))
    rm(list='data_total')
    rm(list='label_total')
    data_total <- NULL
    label_total <-NULL
  }
}

