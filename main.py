import cv2
import face_recognition


imgone=face_recognition.load_image_file('img/known/Michael Jordan.jpg')
imgone=cv2.cvtColor(imgone,cv2.COLOR_BGR2RGB)
oneloc=face_recognition.face_locations(imgone)[0]
encodeone = face_recognition.face_encodings(imgone)[0]
cv2.rectangle(imgone,(oneloc[3],oneloc[0]),(oneloc[1],oneloc[2]),(0,255,0),2)


imgtest=face_recognition.load_image_file('img/unknown/bill-gates-4.jpg')
imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)
testloc=face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(testloc[3],testloc[0]),(testloc[1],testloc[2]),(0,255,0),2)

result=face_recognition.compare_faces([encodeone],encodetest)
facedis=face_recognition.face_distance([encodeone],encodetest)

text="result "+ str(result[0]) + " and difference is " + str(round(facedis[0],2)*100)+"%";
print(text)
cv2.rectangle(imgtest, (7, 20), (700, 100), (0,0,0), -1)
cv2.putText(imgtest,text,(10,60),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)



cv2.imshow('OneImg',imgone)
cv2.imshow('testimg',imgtest)
cv2.imwrite('false_input.png', imgone)
cv2.imwrite('fasle_output.png', imgtest)
cv2.waitKey(0)



