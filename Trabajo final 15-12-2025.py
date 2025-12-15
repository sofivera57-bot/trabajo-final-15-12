import pandas as pd 
import numpy  as np #calculos numéricos
import matplotlib.pyplot as plt #graficos
import statsmodels.api as sm #para regresion
import scipy.stats as stats #estadistica

pd.read_csv("tracking_dataset_clean.csv") #leemos el csv
df=pd.read_csv("tracking_dataset_clean.csv") #lo designamos a una dataframe
print (df)

#Info general 
print(df.info()) #info general del df
print(df.head()) #muestra las primeras filas del df
print(df.describe())#resumen numerico.
print(df.isnull().sum())# vemos NaN por columnas.
print(df.columns.tolist())#mostramos los nombres de las columnas como una lista.

#Limpieza
df.columns = df.columns.str.lower().str.replace(" ","_") #pasamos los nombres de las colunas a minusculas, remplazamos los espacios en blanco por un guion.
df = df[(df["edad"] >= 16) & (df["edad"] <= 95)]# si alguien es menor de 16 y mayor de 95 años se elimina la encuesta.
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce") # pasamos la columna fecha a tipo date time - coerce significa que si hay algun error en la conversion se pone NaT (not a time) y continua el proceso.
df = df.drop_duplicates(subset=["encuesta"])# eliminamos filas duplicadas segun la columna encuesta.
df = df.dropna(subset=["fecha"]) #eliminamos filas con Nat en la columna fecha.
df = df.dropna(subset=["imagen_del_candidato"]) #eliminamos filas con Nan en la columna imagen del candidato.
df = df.dropna(subset=["voto"]) #eliminamos filas con Nan en la columna voto.
df = df.dropna(subset=["edad"]) #eliminamos filas con Nan en la columna edad.

df["sexo"] = df["sexo"].str.lower() #pasamos todos los valores de la columna sexo a minusculas
df["sexo"] = df["sexo"].str.strip() #eliminamos espacios en blanco en la columna sexo.

df["voto"] = df["voto"].str.lower() #pasamos todos los valores de la columna voto a minusculas
df["voto"] = df["voto"].str.strip() #eliminamos espacios en blanco en la columna voto.

df["estrato"] = df["estrato"].str.lower() #pasamos todos los valores de la columna estrato a minusculas
df["estrato"] = df["estrato"].str.strip() #eliminamos espacios en blanco en la columna estrato.


#Ponderacion
 
# pesos por sexo
df["sexo_map"] = df["sexo"].map({"masculino": 1, "femenino": 2})#creamos una nueva columna del df "sexo_map". Despues aplicamos la funcion map para que busque cada valor en sexo y lo remplaze segun el diccionario que armamos.
print(df["sexo_map"])
df["sexo_map"].head(20)#Verificamos que sexo contenga los valores numericos
df["sexo_map"].value_counts().sum()#conteo de cuantas veces aparece las categorias de sexo_map, no cuenta los NaN. #(si quiesieramos que paarezcan los NaN (dropna = false))
cuentas = df["sexo_map"].value_counts()
total = cuentas.sum()# total de respuestas válidas

# total masculinos (codificados como 1)
total_masculino = cuentas.get(1, 0)# get para buscar el valor  asociado con la clve 1(numero total de hombres), si no lo encuentra asigna 0 (por seguridad, para evitar error)
porcentaje_masculino = total_masculino / total *100 #porcentaje de hombres de la muestra,/ el conteo por el total de respuestas y multiplicando por 100 a fn de conocer la distribucion muestral masculina.
print("Total respuestas:", total)
print("Total masculino (1):", total_masculino)
print("Porcentaje masculino:", porcentaje_masculino)
porcentaje_poblacionalmasculino = 48.34 #porcentaje poblacional masculino sacado a partir de los datos del censo 2022
peso_masculino = porcentaje_poblacionalmasculino / porcentaje_masculino #factor de ponderacion masculino. indica cuánto se debe "pesar" o corregir cada respuesta masculina para que el grupo masculino en la muestra se ajuste al 48.34% poblacional.
print(peso_masculino)

# total femenino (codificado como 2)
total_femenino = cuentas.get(2,0)
porcentaje_femenino = total_femenino / total *100
print("Total femenino(2)", total_femenino)
print("Porcentaje femenino:", porcentaje_femenino)
porcentaje_poblacionalfemenino = 51.62#porcentaje poblacional femenino sacado a partir de los datos del censo 2022
peso_femenino = porcentaje_poblacionalfemenino / porcentaje_femenino 
print(peso_femenino)

df_pesos_sexo = {
    1: peso_masculino, 
    2: peso_femenino,           #creamos un diccionario con los pesos por sexo
}
print(df_pesos_sexo)

df["peso_sexo"] = df["sexo_map"].map(df_pesos_sexo)#creamos una nueva columna aplicando el diccionario
print(df["peso_sexo"].head()) #visualizamos los primeros valores de la columna peso_sexo

#PONDERACION DE PROVINCIA
# pesos por provincia (estrato)#prov normalizada#antes estrato normalizado
df["provincia_normalizada"] = df["estrato"].astype(str)#normalizamos y limpiamos la columna "prov" anteriormente estarto
df["provincia_normalizada"].unique()

correcciones = {
    "caba": "ciudad autonoma de buenos aires",
    "capital federal": "ciudad autonoma de buenos aires",
    "capital": "ciudad autonoma de buenos aires",
    "CABA": "ciudad autonoma de buenos aires",
    "C.A.B.A.": "ciudad autonoma de buenos aires",
    "bs as": "buenos aires",
    "bs.as.": "buenos aires",
    "bs as.": "buenos aires",
    "tierra del fuego": "tierra del fuego, antartida e islas del atlantico sur"
}
df["provincia_normalizada"] = df["provincia_normalizada"].str.lower().replace(correcciones) #Aplicamos las correcciones usando .replace()

#definimos el diccionario de mapeo
mapa_provincias = ({
"ciudad autonoma de buenos aires": 1,
"buenos aires": 2,
"catamarca": 3,
"chaco": 4,
"chubut": 5,
"cordoba": 6,
"corrientes": 7,
"entre rios": 8,
"formosa": 9,
"jujuy": 10,
"la pampa": 11,
"la rioja": 12,
"mendoza": 13,
"misiones": 14,
"neuquen": 15,
"rio negro": 16,
"salta": 17,
"san juan": 18,
"san luis": 19,
"santa cruz": 20,
"santa fe": 21,
"santiago del estero": 22,
"tierra del fuego, antartida e islas del atlantico sur": 23,
"tucuman": 24
})

df["provincia_mapa"] = df["provincia_normalizada"].map(mapa_provincias).astype("Int64")#armamos una nueva columna, seleccionamos la columna fuente y aplicamos el diccionario con las reglas de conversion (pasa de texto a numero),finalmente, cnvertimos la columna a tipo de dato entero que acepte los nulos "I".Las prov sin dato o no mapeadas quedan como NaN
print(df["provincia_mapa"].head())

#porcentajes poblacionales por provincia segun censo 2022
porcentajes_poblacionales_por_provincia = ({
    1: 6.8, 2: 38.18, 3: 0.93, 4: 2.46, 5: 1.29,
    6: 8.36, 7: 2.64, 8: 3.1, 9: 1.32, 10: 1.76,
    11: 0.78, 12: 0.83, 13: 4.45, 14: 2.78, 15: 1.54,
    16: 1.63, 17: 3.13, 18: 1.78, 19: 1.18, 20: 0.73,
    21: 7.72, 22: 2.31, 23: 0.4, 24: 3.77
})
muestra_por_provincia = df["provincia_mapa"].value_counts(normalize=True) * 100#calculamos la distribucion muestral por provincia. contamos valores unicos y despues las frecuencias se pasan a proporciones(la suma de todos los valores es 1.0)y despues pasamos a porcentaje

poblacion_por_provincia = pd.Series(porcentajes_poblacionales_por_provincia)#pasamos por.pob.por.prov a series para que su indice,(los codigos de las prov) se alineen con la serie de "muestra por prov"
df_pesos = pd.DataFrame({
    "porcentaje_poblacional": poblacion_por_provincia,#Creamos un df y unimos ambas series
    "porcentaje_muestral": muestra_por_provincia
})

df_pesos["factor_peso_provincia"] = df_pesos["porcentaje_poblacional"] / df_pesos["porcentaje_muestral"] #calculamos el factor de peso.
print(df_pesos.head(24))
df["peso_provincia"] = df["provincia_mapa"].map(df_pesos["factor_peso_provincia"]) #aplicamos el peso provincia al df principal.

#PESOS FINALES.
df["peso_final"] = df["peso_sexo"] * df["peso_provincia"] 
print(df["peso_final"].head(20))

#tracking. 
#IMAGEN DEL CANDIDATO.
df = df[(df["imagen_del_candidato"] >= 0) & (df["imagen_del_candidato"] <= 100)]#eliminamos valores invalidos de imagen -0 o +100
df.set_index("fecha", inplace=True) #columna fecha como índice del df
df.sort_index(inplace=True) #ordenamos el data a partir del indice fecha. (inplace=true) controla dónde se guarda el resultado, directamente en el df ori
imagen_ponderada_diaria = df.groupby(df.index)["imagen_del_candidato"].apply(
    lambda x: np.sum(df.loc[x.index, "peso_final"] * x) / np.sum(df.loc[x.index, "peso_final"])
).resample("D").mean() #agrupamos por fecha (índice) y aplicamos la función lambda para calcular la media ponderada diaria de la imagen del candidato. Luego, usamos resample("D") para asegurarnos de que tenemos una entrada para cada día (incluso si no hubo encuestas ese día) y finalmente calculamos la media diaria.
imagen_diaria_interpolada = imagen_ponderada_diaria.interpolate(method="time") #Rellena los vacíos temporales. Toma esos NaN creados por resample y los sustituye con un valor estimado (interpolado).
print(imagen_diaria_interpolada.head())
imagen_rolling = imagen_diaria_interpolada.rolling("7D").mean() #Calcula la media movil de 7 días, aplicada a la serie de media diaria interpolada .Toma todos los valores dentro de los últimos 7 días (no 7 filas, sino 7 días reales en el índice de tiempo) y calcula la media.
print(imagen_rolling.head())
imagen_rolling_std = imagen_ponderada_diaria.rolling("7D", min_periods=3).std() #calculamos el desvio estandar (variabilidad de los datos) dentro de la ventana de 7 dias.
print(imagen_rolling_std.head())
casos_por_ventana = imagen_ponderada_diaria.rolling("7D", min_periods=3).count()# Cantidad de casos reales por ventana de 7 días.
print(casos_por_ventana.head())
error_estandar = imagen_rolling_std / np.sqrt(casos_por_ventana)# Error estandar: se mide a partir del desvio estandar, mide que tan presisa es la estimacion de la media. np.sqrt: raixz cuadrada de las observaciones de cada ventana
print(error_estandar.head())

grados_libertad = casos_por_ventana - 1 # 1. Calcular los Grados de Libertad (df = n - 1). LOS CASOS POR VENTANA SON LOS CASOS REALES QUE TIENE LA VENTANA.(Osea n-1)la t de Student usa df = n - 1 para calcular el percentil crítico.
t_critico = stats.t.ppf(0.975, grados_libertad) # 2. Calcular el Valor Crítico de t (dinámico) El percent point function (ppf) nos da el valor t para un 97.5% de probabilidad (IC 95%).stats.t.ppf(): Es la función de percentil de la distribución $t$ de la librería scipy. 0.975: Es el nivel de probabilidad que deja 2.5% de área en la cola superior (para un IC bilateral del 95%).
# Aplicamos el t_critico para el Intervalo de Confianza
ic_superior = imagen_rolling + t_critico * error_estandar # Suma al valor de la Media Móvil (imagen_rolling) el margen de error (valor t x error estandar). definimos el valor máximo donde se espera que se encuentre la media poblacional el 95% de las veces.
ic_inferior = imagen_rolling - t_critico * error_estandar # resta al valor de la Media Móvil (imagen_rolling) el margen de error (valor t x error estandar). definimos el valor minimo donde se espera que se encuentre la media poblacional el 95% de las veces.

tendencia_imagen = pd.DataFrame({
    "media_movil_diaria":imagen_rolling,
    "desvio_estandar_semanal": imagen_rolling_std
})
resultado_ic = pd.DataFrame({
    "media_semanal": imagen_rolling.round(2),
    "IC_inferior": ic_inferior.round(2),
    "IC_superior": ic_superior.round(2)
})

print(resultado_ic.tail(10))

print(tendencia_imagen.tail(20).round(2))

#GRAFICO DE IMAGEN DE CANDIDATO.
plt.figure(figsize=(14, 7)) #tamaño del grafico


plt.fill_between(  #Banda del intervalo de confianza. Visualiza el rango de incertidumbre estadística (el margen de error) en torno a la media.
    imagen_rolling.index, #eje x:fecha
    ic_inferior, #limite inferior
    ic_superior, #limite superior
    alpha=0.25, #transparencia del la banda de IC
    label="IC 95%" #leyenda
)

plt.plot(    # linea de la media movil
    imagen_rolling.index, #eje x:fecha
    imagen_rolling,   #eje y: valores
    linewidth=2, #grosor de la linea 
    label="Imagen (media móvil 7D)" #leyenda 
)

plt.title("Tracking de Imagen del Candidato (Media Móvil 7D + IC95%)")
plt.xlabel("Fecha")
plt.ylabel("Imagen (%)")
plt.grid(True, linestyle="--", alpha=0.6) #cuadricula de fondo, 0.6 para que sean semitransparentes 
plt.legend()
plt.tight_layout() #ajuste de los parametros de la figura 
plt.show() #visualizacion


#Intencion de voto. 

df["voto"].dropna().unique().tolist()#dropna. eliminamos nan, U. Par devolver valores unicos, T, para verlo como una lista
df["intencion_de_voto"] = df["voto"]#creamos la columna int.de voto (pasamos la info de voto a la nueva columna)
votos_diarios_ponderados = df.groupby(["fecha", "intencion_de_voto"])["peso_final"].sum().unstack(fill_value=0) #agrupamos por fecha e intencion de voto, sumamos los pesos finales para cada grupo y luego usamos unstack para pivotar los candidatos a columnas.
total_por_dia = votos_diarios_ponderados.sum(axis=1) #.sum.función para sumar valores.#axis1:Indica a pandas que realice la operación de suma a lo largo del Eje 1 (el eje de las columnas).
porcentaje_diario = (votos_diarios_ponderados.div(total_por_dia, axis=0) * 100).round(2)#ndica a pandas que realice la división alineando los índices (es decir, las fechas). Osea, para tal dia, se dividen los votos de c/candidato por el total por dia de ese dia.
tracking_7d = porcentaje_diario.rolling(window=7, min_periods=3).mean().round(2)#MEDIA MOVIL DE LOS ULTIMOS 7NDIAS
print(tracking_7d)
candidato = ["candidato a", "candidato b", "candidato c"] #filtramos solo candidatos
tracking_filtrado = tracking_7d[candidato]
tracking_filtrado = tracking_7d[candidato].dropna(how="all") #eliminamos filas donde todos los candidatos son NaN
print(tracking_filtrado)

#grafico de intencion de voto
plt.figure(figsize=(16,6))

for col in tracking_filtrado.columns:
    plt.plot(tracking_filtrado.index, tracking_filtrado[col], label=col)

plt.title("Evolución de intención de voto (tracking 7 días)")
plt.xlabel("Fecha")
plt.ylabel("Porcentaje (%)")
plt.legend(title="Candidatos")
plt.grid(True, alpha=0.3)
plt.show()

# ANALISIS ESTADISTICO
# REGRESION LINEAL SIMPLE (PONDERADA)

# Definir las variables limpias

X = df[["edad"]] #Variable independiente
y = df["imagen_del_candidato"] #Variable dependiente
pesos = df["peso_final"] 

# Limpiar las filas NaN antes de la regresión
df = df.dropna(subset=["edad", "imagen_del_candidato", "peso_final"])
X = df[["edad"]].copy()
y = df["imagen_del_candidato"].copy()
pesos = df["peso_final"].copy()


X= sm.add_constant(X) # Agregar la intercepción
modelo_ponderado = sm.WLS(y, X, weights=pesos).fit() # WLS: Mínimos Cuadrados Ponderados (Weighted Least Squares)#weights=pesos:le decimos al modelo que cada observacion tiene una importancia diferente.
print(modelo_ponderado.summary()) 

edad_para_grafico = X['edad']
valores_predichos = modelo_ponderado.predict(X) # Obtener los valores predichos por el modelo
plt.figure(figsize=(10, 6))
plt.scatter(edad_para_grafico, y, alpha=0.6, color='skyblue', label='Datos de Encuesta')
plt.plot(edad_para_grafico, valores_predichos, color='red', linestyle='-', linewidth=2, label='Línea de Regresión')

plt.title('Imagen del Candidato vs. Edad')
plt.xlabel('Edad')
plt.ylabel('Imagen del Candidato')
plt.legend() # Muestra las etiquetas que definimos antes
plt.grid(True, linestyle='--', alpha=0.7) # Añade una cuadrícula suave
plt.show() # Muestra el gráfico


###Regresion logistica.

df["y_a"] = (df["voto"] == "candidato a").astype(int) #Creamos una nueva serie de pandas que contiene valores booleanos (True o False).Si voto por cand. A, devuelve true, y voto por otro, devuelve false. .astype(int): para que los datos se conviertan en numero entero Cand A(1). Otro(0)
print(df["y_a"].value_counts) #verifica el conteo de 1 y 0. 

variables_X = ["edad"] #variables predictoras (V. independiente)
y = df["y_a"] #variable dependiente (probabilidad de votar al cand A)
df_modelo = df.dropna(subset=variables_X + ["y_a"])# eliminamos nan en variable X y variable Y

X = df_modelo[variables_X] # definimos ambas variables ya limpias.
y = df_modelo["y_a"]

X = sm.add_constant(X) #agregamos la interseccion B0, el valor de Y cuando las variables X son cero
pesos_modelo = df_modelo["peso_final"]
modelo = sm.Logit(y, X, weights=pesos_modelo).fit() #Logit: regresion logistica# .fit() Una vez que el modelo esta definido con sus variables y pesos se le pide que encuentre los mejores parametros, esdecir, fit aplica el estimador de maxima verosimilitud.

print(modelo.summary()) #mostramos el informe completo del modelo.

#grafico para regresion logistica
#robabilidad de votar al candidato A según edad 
edad = np.linspace(df["edad"].min(), df["edad"].max(), 100)

X_pred = pd.DataFrame({
    "const": 1,
    "edad": edad_grid
})

prob = modelo.predict(X_pred)

plt.plot(edad_grid, prob)
plt.xlabel("Edad")
plt.ylabel("Probabilidad de votar al Candidato A")
plt.title("Regresión logística")
plt.show()
 
#TEST DE HIPOTESIS MANN-WHITNEY

#objetivo: Determinar si la distribución de la variable "imagen del candidato" en una población es diferente a la distribución de la misma variable en la otra población

#H0: No hay diferencias significativas entre hombres y mujeres en la imagen del candidato.
#H1: Hay diferencias significativas entre hombres y mujeres en la imagen del candidato.

from scipy.stats import mannwhitneyu

grupo_h = df[df["sexo_map"] == 1]["imagen_del_candidato"].dropna() #Selecciona las filas donde sexo map es 1. Seleccionamos la columna de interes #im del cand" y eliminamos cualquier valor nulo que pueda existir en la imagen del candidato para este grupo
grupo_m = df[df["sexo_map"] == 2]["imagen_del_candidato"].dropna() #Seleccionamos las filas donde sexo map es 2
print("Hombres:",grupo_h) #verificacion de casos efectivo.
print("Mujeres:",grupo_m)


estadistico_U, p_valor = mannwhitneyu(grupo_h, grupo_m, alternative="two-sided") #T. Mann-Whitney U. alternative="two-sided": indica que el test busca diferencias en ambas direcciones (es decir, la imagen de Hombres puede ser mayor o menor que la de Mujeres).
print("valor de U", estadistico_U)#valor estadistico y p-valor
print("p-value:", p_valor)
if p_valor < 0.05: #decision estadistica 
    print("Rechazamos la hipótesis nula H0: Hay diferencias significativas entre hombres y mujeres en la imagen del candidato.")
else:
    print("No rechazamos la hipótesis nula H0: No hay diferencias significativas entre hombres y mujeres en la imagen del candidato.")




