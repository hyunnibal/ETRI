setwd('./Github/ETRI/data')
data_total<-NULL
label_total<-NULL

for(k in 0:146){
  #데이터 불러오기
  data<-read.csv('./vitalPeriodic.csv',nrow=1000000,stringsAsFactors = FALSE, na.strings = 'NA',skip=999000*k)
  #환자 목록
  patient<-unique(data[,1])
  
  for(i in 1:length(patient)){
    #해당 환자 데이터
    data_p = data[grep(patient[i],data[,1]),]
    sorted_data_p = data_p[order(data_p[,6]),]
    
    #너무 늦지않게 측정을 시작했는가? 결손치가 너무 많은가?
    if(sorted_data_p[1,6]<15&&sum(is.na(sorted_data_p[1:60,15]))<45){
      #결손치 체크
      if(nrow(sorted_data_p)>60 && sorted_data_p[60,6]==sorted_data_p[1,6]+295){
        data_c = sorted_data_p[1:60,c(1,6,8,9,10,15)]
        x=data_c[1,1]
        for(j in 1:20){
          #데이터 압축
          temp<-data_c[(j*3-2):(j*3),3:6]
          x<-cbind(x,median(temp[,1],na.rm=T),median(temp[,2],na.rm=T),median(temp[,3],na.rm=T),median(temp[,4],na.rm=T))
        }
        #압축 결과 결손치 있으면 제거
        if(sum(is.na(x)) == 0){
          #라벨
          label=cbind(as.logical(sum(x[seq(66,81,4)]<95)),as.logical(sum(x[seq(67,81,4)]<70)),as.logical(sum(x[seq(67,81,4)]>100)),as.logical(sum(x[seq(68,81,4)]<13)),as.logical(sum(x[seq(68,81,4)]>20)),as.logical(sum(x[seq(69,81,4)]<70)),as.logical(sum(x[seq(69,81,4)]>110)))
          data_total<-rbind(data_total,x[1:65])
          label_total<-rbind(label_total,label)
        }
      }
    }
  }
  message('recent k = ',k)
  #메모리 초기화
  rm(list='data')
  rm(list='data_p')
  rm(list='data_c')
  rm(list='sorted_data_p')
  #저장
  if(k%%10==0 | k==146){
    write.csv(data_total,file=sprintf("./data_total%d.csv",k))
    write.csv(label_total,file=sprintf("./label_total%d.csv",k))
    rm(list='data_total')
    rm(list='label_total')
    data_total <- NULL
    label_total <-NULL
  }
}

#데이터 통합
data_idx<-grep('data',dir('./'))
label_idx<-grep('label',dir('./'))

data_total = NULL
for(i in data_idx){
  data_total = rbind(data_total, read.csv(dir[i],sep = ',', header = TRUE))
  print(dim(data_total))
}

label_total = NULL
for(i in label_idx){
  label_total = rbind(label_total, read.csv(dir[i],sep = ',', header = TRUE))
  print(dim(label_total))
}

write.csv(data_total[,-1],file='eicu_data.csv')
write.csv(data.matrix(label_total[,-1]),file='eicu_label.csv')