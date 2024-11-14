import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt

data = pandas.read_csv('RegressionData.csv', header = None, names=['X', 'y']) 

X = data['X'].values.reshape(-1,1) 
y = data['y'].values 
plt.scatter(X, y) 
reg = linear_model.LinearRegression() 
reg.fit(X, y) 


fig = plt.figure()
y_pred = reg.predict(X) 
plt.scatter(X,y, c='b') 
plt.plot(X, y_pred, 'r') 
fig.canvas.draw()
print("The linear relationship between X and y was modeled according to the equation: y = b_0 + X*b_1, \
where the bias parameter b_0 is equal to ", reg.intercept_, " and the weight b_1 is equal to ", reg.coef_[0])
print("the profit/loss in a city with 18 habitants is ", reg.predict([[18]])[0])

data = pandas.read_csv('LogisticRegressionData.csv', header = None, names=['Score1', 'Score2', 'y']) 

X = data[['Score1', 'Score2']] 
y = data['y'] 

m = ['o', 'x']
c = ['hotpink', '#88c999']
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i], marker=m[data['y'][i]], color = c[data['y'][i]]) 
fig.canvas.draw()

regS = linear_model.LogisticRegression() 
regS.fit(X, y)

y_pred = regS.predict(X)
m = ['o', 'x']
c = ['red', 'blue']
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i], marker=m[y_pred[i]], c = c[y_pred[i]])
fig.canvas.draw()