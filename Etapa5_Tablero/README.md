El tablero fue construido con Dash + Plotly + Bootstrap, siguiendo un estilo limpio y profesional.

El tablero permite:

A. Visualización
	•	Histograma de precios.
	•	Boxplot de precios por tipo de habitación.
	•	Gráfico de dispersión precio vs capacidad.

B. Interacción con los modelos

A través de un panel en la izquierda, el usuario puede:
	•	Seleccionar qué modelo quiere utilizar:
	•	Regresión NN
	•	Regresión TEC
	•	Clasificación NN
	•	Clasificación TEC
	•	El tablero genera automáticamente inputs dinámicos según las variables necesarias del modelo.
	•	El botón “Calcular” devuelve:
	•	Precio estimado (modelos de regresión)
	•	“RECOMENDADO / NO RECOMENDADO” y probabilidad (modelos de clasificación)

