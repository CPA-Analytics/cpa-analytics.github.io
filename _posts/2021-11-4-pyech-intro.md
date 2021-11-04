---
layout: post
title: Introducción a PyECH
date: 2021-11-04
tags: oss open source
---

> Procesamiento de la ECH del INE en Python

*TL;DR*: [https://cpa-analytics.github.io/pyech](cpa-analytics.github.io/pyech) para uso básico mediante una interfaz gráfica, o [https://github.com/cpa-analytics/pyech](https://github.com/cpa-analytics/pyech) para uso avanzado.

----

La Encuesta Continua de Hogares (ECH) del Instituto Nacional de Estadística es la fuente más importante de información socieconómica de las personas y hogares en Uruguay, cubriendo dimensiones que van desde el nivel educativo de las personas hasta la calidad de la vivienda en la que viven. Existe desde por lo menos 1981, pero su versión actual permanece razonablemente estable desde 2006<sup>1</sup>. Desde entonces, la ECH tiene unas 400-500 variables y unos 120-150 mil registros.

Trabajar con la Encuesta Continua de Hogares del Instituto Nacional de Estadística no es un asunto trivial. Algunas de las dificultades son:

1. Los microdatos de las encuestas históricamente se presentaron en formatos propietarios. Aunque en general se pueden encontrar también formatos DBF y DAT, el único formato disponible que contiene información de etiquetas es SAV, que requiere el software SPSS de IBM<sup>2</sup>. La encuesta 2020 incorpora el formato RDATA para el lenguaje R (libre).
2. Aún contando con el software necesario, existen ciertas limitantes que hacen poco fluido su uso; por ejemplo, hasta hace 2 años STATA (en general de uso más difundido que SPSS en Uruguay) no podía mantener dos datasets en la memoria de manera simultánea, lo que implica constantemente tener que cargar la encuesta para comparar resultados entre años.
3. Por ser una encuesta y no un censo, es necesario usar ponderadores para obtener estimaciones a partir de los datos.
4. Los nombres de variables cambian a lo largo de los años y es difícil saber a qué refiere cada una.

PyECH es una librería escrita en Python que apunta a facilitar algunas de estas cosas. En el fondo es un proyecto simple que pretender hacer 3 ó 4 cosas muy bien:

* Descargar y descomprimir archivos de encuestas.
* Leer los datos de la encuesta correspondiente.
* Obtener el diccionario de variables y procesarlo.
* Calcular agregados y cruzar variables.

Esto quiere decir que, por ejemplo, calcular la cantidad de desocupados por sexo y departamento en 2013 se vuelve razonablemente simple<sup>3</sup>.

Pero lo más interesante de que PyECH esté escrita en un lenguaje de código abierto es que las licencias de software se vuelven innecesarias, y que disponibilizarla para que otros la puedan usar es simple. En ese sentido, la librería puede ser usada como [un módulo más en Python](https://pypi.org/project/pyech/), dentro de [Google Colab](https://colab.research.google.com/github/CPA-Analytics/pyech/blob/master/examples/example.ipynb) y otras soluciones online de notebooks, que permiten escribir y ejecutar Python sin descargar ni instalar nada, o en una [interfaz gráfica](https://cpa-analytics.github.io/pyech/) construida para mostrar su uso básico.

----
1. [La ECH 2020 y posteriores representan un cambio frente a las de 2006-2019](https://www.ine.gub.uy/c/document_library/get_file?uuid=359cba03-b448-400f-9e5b-10136bdfb519&groupId=10181).
2. Esto no es estrictamente cierto, como el propio PyECH demuestra. Es decir, es posible leer archivos SAV sin SPSS, lo que no implica que sea fácil.
3. ```python
   from pyech import ECH
   
   survey = ECH()
   survey.load(2013, weights="pesoano") # Descarga la ECH 2013 desde la web del INE, descomprime el .rar y lee el .sav correspondiente a hogares y personas.
   survey.summarize("pobpcoac", by=["e26", "dpto"], aggfunc="count")
   ```
