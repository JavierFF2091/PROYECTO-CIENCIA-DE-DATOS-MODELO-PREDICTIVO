#El principal objetivo es construir un modelo predictivo que pueda predecir el monto total de una transacción (TotalAmount) en un entorno minorista en función de las otras variables disponibles en el conjunto de datos. Este modelo ayudará a entender mejor los factores que influyen en el monto total de una transacción y proporcionará una herramienta útil para predecir el monto esperado de las futuras transacciones.
#Objetivos expecificos:
#Predecir el monto total de una transacción en función de variables como la cantidad de productos comprados, el precio unitario, el descuento aplicado, etc.
#Identificar las variables más influyentes en el monto total de una transacción.
#Proporcionar información valiosa sobre el comportamiento de compra de los clientes y las tendencias de ventas.
#Utilizar el modelo para optimizar estrategias de fijación de precios, descuentos y gestión de inventario.
#Mejorar la experiencia del cliente al comprender mejor sus patrones de compra y preferencias.

#Fuente de datos: Kaggle

#Descripción de las variables.
#Variable Target: Total Amount
#Descripción: La cantidad total de la transacción en términos monetarios.
#Tipo: Variable numérica continua.
#Objetivo: Predecir este valor a partir de otras variables para entender los factores que influyen en el monto total de compra de los clientes.
#Predictores (Variables Independientes):
#Customer ID: Identificación única del cliente.
#Product ID: Identificación única del producto comprado.
#Quantity: La cantidad de unidades de un producto compradas en la transacción.
#Price: Precio unitario de cada producto.
#Transaction Date: Fecha y hora de la transacción.
#Payment Method: Método de pago utilizado para la transacción.
#Store Location: Ubicación de la tienda donde se realizó la transacción.
#Product Category: Categoría del producto comprado.
#Discount Applied (%): El porcentaje de descuento aplicado a la transacción.

#Carga de liberías

library("tidymodels")
library(dplyr)
library(fitdistrplus)
library(ggplot2)
library(reshape2)
library(gamlss)
library(gamlss.dist)
library(gamlss.add)
library(childsds)
library(forecast)
library(tseries)
library(readr)

set.seed(123)

#Carga de la DATA

Data <- read.csv(file.choose())
View(Data)

#Hacemos modificaciones en los nombres de las columnas

print(names(Data))
names(Data) = c( "Customer_ID", "Product_ID", "Quantity", "Price", "Transaction_Date", "Payment_Method", "Store_Location", "Product_Category", "Discount_Applied", "Total_Amount")
print(names(Data))
View(Data)

#Redondeamos los datos numéricos a dos decimales

Data$Price <- round(Data$Price, 2)

Data$`Total_Amount` <- round(Data$`Total_Amount`, 2)

Data$`Discount_Applied` <- round(Data$`Discount_Applied`, 2)

View(Data)

#Vamos a convertir las variables categóricas en numéricas utilizando la codificación one-hot

#Variable categórica Product ID
#Primero, convierto 'Product ID' en un factor

Data$`Product_ID` <- factor(Data$`Product_ID`)

#Segundo, asigno un número único a cada categoría

levels(Data$`Product_ID`) <- c(1, 2, 3, 4)  #Asignamos los números 1, 2, 3, 4 a las categorías A, B, C, D respectivamente

#Finalmente, convierto los niveles del factor en números
Data$`Product_ID` <- as.numeric(Data$`Product_ID`)

#Variable categórica Payment Method
#En primer lugar, convierto 'Payment Method' en un factor
Data$`Payment_Method` <- factor(Data$`Payment_Method`, levels = c("PayPal", "Credit Card", "Cash", "Debit Card"))

#Finalmente, convierto los niveles del factor en números
Data$`Payment_Method` <- as.numeric(Data$`Payment_Method`)

#Variable categórica Product Category
#Primero, convierto 'Product Category' en un factor

Data$`Product_Category` <- factor(Data$`Product_Category`, levels = c("Clothing", "Books", "Electronics", "Home Decor"))

#Segundo, asigno un número único a cada categoría

levels(Data$`Product_Category`) <- c(1, 2, 3, 4)  #Asignamos los números 1, 2, 3, 4

#Finalmente, convierto los niveles del factor en números
Data$`Product_Category` <- as.numeric(Data$`Product_Category`)
View(Data)

#Transformación y limpiaza de la columna Transaction Date

Data$`Transaction_Date` <- as.POSIXct(Data$`Transaction_Date`, format = "%m/%d/%Y %H:%M")

#Verificamos rangos de fechas y horas
summary(Data$`Transaction_Date`)

#Eliminar filas con fechas faltantes
Data <- Data[!is.na(Data$`Transaction_Date`), ]
View(Data)

#Comprobamos la limpieza de los datos
str(Data)
colSums(is.na(Data))


#Análisis exploratorio y Data Wrangling

#Estadísticas descriptivas
summary(Data)
quantile(Data$`Total_Amount`, c(0.25, 0.50, 0.75))
table(Data$`Product_Category`)
range(Data$Price)
IQR(Data$Quantity)
var(Data$`Total_Amount`)

#Visualización de datos
#Gráfico de correlación
pairs(Data[, c("Product_ID", "Quantity", "Price", "Payment_Method", "Product_Category", "Discount_Applied", "Total_Amount")])

#Gráfico de dispersión múltiple
plot(Data$Quantity, Data$Price, xlab = "Cantidad", ylab = "Precio", main = "Gráfico de Dispersión de Cantidad vs. Precio")

#Histograma de la variable TotalAmount
hist(Data$`Total_Amount`)

#Gráfico de barras para la variable Product Category
barplot(table(Data$`Product_Category`))

#Diagrama de dispersión para investigar la relación entre Precio y TotalAmount
plot(Data$Price, Data$`Total_Amount`)

#Consulta de datos faltantes
colSums(is.na(Data))

#Identificación de valores atípicos
boxplot(Data$Price, main = "Diagrama de Caja del Precio", ylab = "Precio")

#Análisis de correlación 
library(corrplot)
cor_matrix <- cor(Data[, c("Product_ID", "Quantity", "Price", "Payment_Method", "Product_Category", "Discount_Applied", "Total_Amount")])
corrplot::corrplot(cor_matrix, method = "color")

#La cantidad de productos comprados (Quantity) está fuertemente correlacionada con el monto total de la transacción (Total Amount), con una correlación de 0.69. Esto sugiere que a medida que aumenta la cantidad de productos comprados, el monto total de la transacción tiende a aumentar.
#El precio unitario de los productos (Price) también está correlacionado positivamente con el monto total de la transacción (Total Amount), con una correlación de 0.636. Esto indica que a medida que aumenta el precio unitario de los productos, el monto total de la transacción tiende a aumentar.


#Segmentación de los datos

#Filtramos las variables relevantes para la segmentación
data_segmentation <- Data[, c("Product_Category", "Total_Amount", "Discount_Applied", "Payment_Method", "Product_ID", "Quantity", "Price")]

#Normalizar los datos
data_normalized <- scale(data_segmentation[, -1])  #Excluyendo la variable categórica

#Realizar el clustering utilizando K-means
set.seed(123)  
kmeans_model <- kmeans(data_normalized, centers = 4)  

#Asignar los segmentos a cada observación en los datos
data_segmentation$Segment <- as.factor(kmeans_model$cluster)

#Visualizar los segmentos
library(ggplot2)
ggplot(data_segmentation, aes(x = Quantity, y = Price, color = Segment)) +
  geom_point() +
  labs(title = "Segmentación basada en Categoría de Producto", x = "Cantidad", y = "Precio") +
  theme_minimal()


#Crear un recipe
library(recipes)
library(magrittr)
install.packages("magrittr")

set.seed(123)

#Creamos el ricpe
preprocess_rec <- recipe(`Total_Amount` ~ ., data = data_segmentation)

#Normalización de variables numéricas
preprocess_rec <- step_normalize(preprocess_rec, all_numeric(), -all_outcomes())

#Paso de codificación one-hot de variables categóricas
preprocess_rec <- step_dummy(preprocess_rec, all_nominal(), -all_outcomes())

#Finalmente, preparamos para aplicar la receta a los datos
prepared_data <- prep(preprocess_rec, training = data_segmentation)


#Cargar la librería para regresión lineal
library("stats")
set.seed(123)

#Creamos un modelo de regresión lineal múltiple
lm_model <- lm(`Total_Amount` ~ ., data = data_segmentation)

#Ver el resumen del modelo
summary(lm_model)


#Cargamos la librería para Random Forest
library(randomForest)
library(caret)
set.seed(123)

#Convertimos las variables categoricas como factor
data_segmentation$Product_Category <- factor(data_segmentation$Product_Category)
str(data_segmentation)

#Creamos un modelo de Bosque Aleatorio
rf_model <- randomForest(`Total_Amount` ~ ., data = data_segmentation, ntree = 50)

#Ver el resumen del modelo
summary(rf_model)
print(rf_model)


#Creación de Workflows
library(recipes)
library(glmnet)
library(workflows)
library(magrittr)
library(tune)
set.seed(123)

#Definimos la receta de preprocesamiento
recipe_lm <- recipe(`Total_Amount` ~ ., data = data_segmentation) %>%
  step_normalize(all_numeric()) %>%
  step_dummy(all_nominal())

#Combinamos la receta y el modelo en un workflow
workflow_lm <- workflow() %>%
  add_recipe(recipe_lm) %>%
  add_model(lm_model)

#Ajustamos el workflow a los datos
trained_workflow_lm <- workflow_lm %>% 
  fit(data = data_segmentation)

#Especificamos la configuración para tunear los hiperparámetros de Regresión Lineal Múltiple
tune_spec_lm <- linear_reg(mode = "regression") %>%
  finalize_workflow() %>%
  workflow_set_engine("caret")

#Realizamos la búsqueda de hiperparámetros para Regresión Lineal Múltiple
tuned_lm <- tune(grid = 5, workflow = workflow_lm, resamples = data_segmentation) %>%
  collect_metrics()

#Ver los resultados de Regresión Lineal Múltiple
summary(tuned_lm)

library(recipes)
library(randomForest)
library(tune)

#Definimos la receta de preprocesamiento
recipe_rf <- recipe(`Total_Amount` ~ ., data = data_segmentation) %>%
  step_normalize(all_numeric()) %>%
  step_dummy(all_nominal())

#Combinamos la receta y el modelo en un workflow
workflow_rf <- workflow() %>%
  add_recipe(recipe_rf) %>%
  add_model(rf_model)

#Ajustamos el workflow a los datos
trained_workflow_rf <- workflow_rf %>% 
  fit(data = data_segmentation)

#Especificamos la configuración para tunear los hiperparámetros de Random Forest
tune_spec_rf <- rand_forest(mode = "regression") %>%
  finalize_workflow() %>%
  workflow_set_engine("caret")

#Realizamos la búsqueda de hiperparámetros para Random Forest
tuned_rf <- tune(grid = 5, workflow = workflow_rf, resamples = data_segmentation) %>%
  collect_metrics()

#Ver los resultados de Random Forest
summary(tuned_rf)


#Elegimos el objeto workflow que presenta mejores métricas.
#Random Forest

set.seed(123)

#Obtenemos los mejores hiperparámetros para Random Forest
best_params_rf <- select_best(tuned_rf, "rmse")

#Ajustamos el workflow con los mejores hiperparámetros
best_workflow_rf <- finalize_workflow(workflow_rf, best_params_rf)

#Ajustamos el workflow a los datos completos
final_fit_rf <- fit(best_workflow_rf, data = data_segmentation)

#Evaluamos el rendimiento del modelo final
final_perf_rf <- collect_metrics(final_fit_rf)
summary(final_perf_rf)

#Regresión lineal Múltiple

set.seed(123)

#Obtenemos los mejores hiperparámetros para Regresión Lineal Múltiple
best_params_lm <- select_best(tuned_lm, "rmse")

#Ajustamos el workflow con los mejores hiperparámetros
best_workflow_lm <- finalize_workflow(workflow_lm, best_params_lm)

#Ajustamos el workflow a los datos completos
final_fit_lm <- fit(best_workflow_lm, data = data_segmentation)

#Evaluamos el rendimiento del modelo final
final_perf_lm <- collect_metrics(final_fit_lm)
summary(final_perf_lm)



