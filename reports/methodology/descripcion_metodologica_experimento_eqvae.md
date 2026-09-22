---
title: "Descripción metodológica reproducible del experimento EqVAE sobre UBC-OCEAN"
subtitle: "Documento autónomo para reconstruir el flujo de datos, los modelos y los análisis"
lang: es
date: 2026-09-20
date-format: long
toc: true
toc-title: "Contenido"
toc-depth: 3
number-sections: true
format:
  pdf:
    documentclass: article
    pdf-engine: pdflatex
    papersize: letter
    fontsize: 10pt
    geometry:
      - top=2.2cm
      - bottom=2.2cm
      - left=2.35cm
      - right=2.35cm
    colorlinks: true
    linkcolor: MidnightBlue
    urlcolor: MidnightBlue
    citecolor: MidnightBlue
    code-overflow: wrap
    keep-tex: false
    include-in-header:
      text: |
        \usepackage{microtype}
        \usepackage{booktabs}
        \usepackage{longtable}
        \usepackage{xcolor}
        \definecolor{MidnightBlue}{RGB}{18,55,92}
        \setlength{\parindent}{0pt}
        \setlength{\parskip}{0.55em}
        \widowpenalty=10000
        \clubpenalty=10000
        \displaywidowpenalty=10000
---

# Finalidad y forma de uso

Este documento describe, de forma autónoma, el experimento realizado para comparar dos autoencodificadores variacionales aplicados a parches histopatológicos de UBC-OCEAN: un VAE convolucional convencional y un VAE con convoluciones direccionables equivariantes al grupo continuo de rotaciones planas $\mathrm{SO}(2)$. El objetivo es que una persona con experiencia en PyTorch, o una inteligencia artificial capaz de programar en PyTorch, pueda reconstruir un experimento metodológicamente equivalente sin consultar el historial del proyecto.

El texto conserva las decisiones que definen el experimento: poblaciones de datos, criterios de selección, formas tensoriales, ecuaciones, operaciones, pérdidas, protocolos de evaluación e invariantes científicos. Se omiten decisiones accidentales de ejecución, como el número o modelo de GPU, la cantidad de archivos físicos utilizados, los nombres de sesiones, los comandos empleados y los mecanismos de recuperación. Cuando fue necesario usar aceleración por GPU, la ejecución se realizó en Kaggle.

La descripción diferencia cuatro niveles:

1. **Contrato científico:** qué pregunta se quiso responder y qué debía mantenerse comparable.
2. **Contrato de datos:** qué representa cada fila, parche, WSI y tensor.
3. **Contrato matemático:** qué operaciones debe realizar una implementación equivalente.
4. **Estado experimental:** qué etapas terminaron y cuáles continúan en desarrollo.

No debe interpretarse este documento como una afirmación de superioridad de una arquitectura. La comparación cambia la parametrización, el número de parámetros libres y el costo de las convoluciones, aunque mantiene iguales las poblaciones, las formas de entrada y latente, el presupuesto de entrenamiento y los protocolos de evaluación.

## Estado de las etapas

| Etapa | Estado al 20 de septiembre de 2026 | Alcance |
|---|---|---|
| Construcción de atlas y poblaciones | Completada | Selección de tejido, máscaras, particiones y vistas lógicas |
| Entrenamiento de los dos VAE | Completado y congelado | Ambos modelos se fijaron después de 60 000 actualizaciones |
| Reconstrucción y tareas supervisadas | Completadas | Reconstrucción, diagnóstico de WSI y tejido por parche |
| PCA espacial y sondeos de rotación | Completados | Visualizaciones, controles exactos y barridos angulares |
| Geometría funcional, etapa A1 | Completada | Acción exacta de $C_4$ y espectro local del decodificador |
| Caminos funcionales, etapa A2 | Primera ejecución completada | Los caminos se calcularon, pero el presupuesto de optimización resultó insuficiente |
| Continuación A2 | En desarrollo | Continuar los mismos estados hasta estabilizar energía y gradiente |
| Refinamientos posteriores | Planeados y condicionados | Mayor resolución temporal, búsqueda de fibras, órbita continua, disparo y holonomía |

# Visión general del experimento

El flujo completo fue:

$$
\text{WSI}
\longrightarrow
\text{selección de tejido}
\longrightarrow
\text{atlas de coordenadas}
\longrightarrow
\text{parches}
\longrightarrow
\text{VAE}
\longrightarrow
\boldsymbol\mu
\longrightarrow
\begin{cases}
\text{reconstrucción},\\
\text{clasificación de WSI},\\
\text{clasificación de tejido},\\
\text{sondeos del espacio latente}.
\end{cases}
$$

La unidad biológica usada para separar poblaciones fue la WSI, no el parche. Esto evita que regiones vecinas de una misma lámina aparezcan simultáneamente en entrenamiento y evaluación. Los parches se usaron como unidades computacionales; cuando una afirmación correspondía a una tarea de WSI, la incertidumbre se calculó agrupando por WSI.

Los dos VAE recibieron los mismos parches, normalización, corrupción, objetivo, presupuesto de actualizaciones y acceso a validación. Ambos producen un posterior gaussiano espacial con media y log-varianza de forma

$$
(B,16,32,32),
$$

para entradas RGB de forma

$$
(B,3,256,256).
$$

Después del entrenamiento se congelaron los dos modelos. Las tareas posteriores utilizaron la media posterior $\boldsymbol\mu$ como representación determinista. No se ajustaron nuevamente los VAE usando las etiquetas de diagnóstico o tejido.

# Construcción de los conjuntos de datos

## Fuente y unidad experimental

La fuente fue UBC-OCEAN, que contiene WSI de carcinoma ovárico y etiquetas de subtipo: carcinoma de células claras (CC), endometrioide (EC), seroso de alto grado (HGSC), seroso de bajo grado (LGSC) y mucinoso (MC). Las TMA se excluyeron porque su escala y organización física son diferentes de las WSI convencionales.

Se utilizaron dos poblaciones sin solapamiento:

- **Desarrollo de los VAE:** 322 WSI para entrenamiento y 39 WSI para validación, representadas por 300 000 y 30 000 parches, respectivamente.
- **Evaluaciones posteriores:** 152 WSI no TMA con máscaras suplementarias. Estas se dividieron en 106 WSI de entrenamiento supervisado, 23 de validación y 23 de prueba sellada.

La segunda cohorte permaneció fuera del entrenamiento de los VAE. Su partición se fijó antes de inspeccionar representaciones o predicciones. El diagnóstico y la disponibilidad de tejido anotado se utilizaron para obtener una partición factible, pero los píxeles, latentes y resultados de los modelos no participaron en esa decisión.

## Cuadrícula de parches

Cada WSI se representó mediante una cuadrícula regular de parches RGB no solapados de $256\times256$ píxeles. Si $(x,y)$ es la esquina superior izquierda de un parche, solo se aceptaron posiciones para las cuales el rectángulo completo estuviera dentro de los límites reales de la WSI.

La cuadrícula cumple dos funciones:

1. define una identidad espacial reproducible para cada parche;
2. permite devolver una predicción de tejido a la lámina como una segmentación dispersa a resolución de parche.

La grilla no debe interpretarse como una segmentación a nivel de píxel. Una celda representa una decisión para una región de $256\times256$ píxeles.

## Selección de primer plano a partir de miniaturas

Leer todas las posiciones de una WSI de gran tamaño es innecesario porque una fracción importante corresponde a fondo. Para identificar tejido sin depender de las anotaciones humanas, se usó la miniatura asociada a cada WSI.

Sea $I_t$ la miniatura RGB. Se convirtió a HSV y se tomó su canal de saturación $S_t$. Sobre este canal se calculó el umbral global de Otsu $\tau$ y se definió una máscara de primer plano

$$
M_t(u,v)=\mathbb 1[S_t(u,v)>\tau].
$$

Para cada parche de la cuadrícula de resolución completa se proyectó su rectángulo sobre la miniatura. Si $A_{xy}$ es el conjunto de píxeles de miniatura cubiertos por esa proyección, su fracción de primer plano fue

$$
f_{\mathrm{fg}}(x,y)
=\frac{1}{|A_{xy}|}\sum_{(u,v)\in A_{xy}}M_t(u,v).
$$

El parche se consideró candidato de tejido cuando

$$
f_{\mathrm{fg}}(x,y)>0.60.
$$

La intención de este paso es separar tejido y fondo de forma independiente de las máscaras. Por ello, las poblaciones de reconstrucción y diagnóstico no quedan guiadas por la disponibilidad de anotación.

## Integración de máscaras suplementarias

Las máscaras suplementarias son imágenes RGB no exhaustivas. Se interpretaron los canales dominantes como:

- rojo: tumor;
- verde: estroma;
- azul: necrosis;
- negro: región no anotada.

El negro no significa tejido normal, fondo ni ausencia de lesión. Solo indica que la fuente no proporcionó una anotación para ese píxel.

Para cada parche se calcularon las fracciones respecto del área completa del parche:

$$
f_T=\frac{n_T}{256^2},\qquad
f_E=\frac{n_E}{256^2},\qquad
f_N=\frac{n_N}{256^2},
$$

donde $n_T,n_E,n_N$ son los píxeles asignados a tumor, estroma y necrosis. La cobertura anotada fue

$$
f_A=\frac{|\{p:\text{el píxel }p\text{ tiene alguna anotación}\}|}{256^2}.
$$

Si un píxel de borde tenía empate en el canal dominante, podía contribuir a más de una fracción de clase, mientras que $f_A$ lo contaba una sola vez. Esto evita perder bordes antialias de las máscaras.

La clase dominante fue la de mayor fracción. Para la tarea de tejido se exigió:

$$
f_A\geq0.10,
\qquad
\frac{\max(f_T,f_E,f_N)}{f_A}\geq0.95.
$$

El primer criterio descarta contactos diminutos con una anotación. El segundo selecciona regiones de alta pureza y excluye transiciones entre tejidos. La tarea resultante evalúa clasificación de tejido en regiones anotadas de alta pureza; no evalúa toda la WSI.

## Atlas de coordenadas

El atlas se construyó como la unión sin duplicados de:

1. todas las celdas que cumplían el criterio de primer plano de Otsu;
2. todas las celdas que intersectaban al menos un píxel de máscara.

Cada fila conservó, como mínimo:

| Campo | Significado |
|---|---|
| `wsi_id` | identidad de la lámina |
| `x`, `y` | esquina superior izquierda del parche |
| `diagnosis_label` | subtipo asignado a la WSI |
| `selection_source` | Otsu, máscara o ambos |
| `annotated_fraction` | fracción anotada del parche |
| `tumor_fraction` | fracción de tumor |
| `stroma_fraction` | fracción de estroma |
| `necrosis_fraction` | fracción de necrosis |

Las filas se ordenaron por

$$
(\text{wsi\_id},y,x).
$$

Este orden tiene una intención metodológica y computacional: los parches consecutivos pertenecen primero a la misma WSI y luego a filas espaciales próximas. Una implementación puede abrir una WSI, recorrerla de arriba hacia abajo y reutilizar regiones cercanas sin saltar continuamente entre láminas o posiciones lejanas. El atlas, y no el orden físico accidental de los archivos, es la autoridad sobre la identidad de cada observación.

## Almacenamiento binario

Los datos se guardaron en formato binario porque cada registro tiene forma y tipo fijos. Esto permite calcular directamente dónde se encuentra una observación y evita decodificar formatos de imagen comprimidos en cada época.

Se distinguieron dos contratos lógicos:

- **Parches RGB:** registros `uint8` con forma `[3,256,256]`, en orden CHW.
- **Representaciones latentes:** registros `float32` con forma `[16,32,32]`, correspondientes a la media posterior $\boldsymbol\mu$.

Un archivo tabular mantiene la correspondencia entre el índice del registro y $(\text{wsi\_id},x,y)$. El número de archivos físicos no forma parte del método científico. Una reimplementación puede usar uno o varios archivos siempre que preserve tipo, forma, identidad, orden y ausencia de duplicados.

El propósito de esta decisión es doble:

1. permitir acceso rápido por índice durante entrenamiento e inferencia;
2. separar la representación física de las distintas vistas experimentales.

## Dataset a nivel de parche y a nivel de WSI

El atlas generó dos interfaces principales.

### Interfaz de parche

Devuelve una observación individual:

$$
(\mathbf x_i,\;\text{wsi}_i,\;x_i,\;y_i,\;\text{etiqueta opcional}).
$$

Se utilizó para entrenar los VAE, evaluar reconstrucción y entrenar el clasificador de tejido.

### Interfaz de WSI

Agrupa todos los registros seleccionados de una lámina:

$$
\mathcal B_w=
\{(\boldsymbol\mu_i,x_i,y_i):\text{wsi}_i=w\},
$$

donde el tamaño $N_w=|\mathcal B_w|$ cambia entre WSI. Esta interfaz se utilizó para clasificación de diagnóstico mediante aprendizaje de instancias múltiples (MIL).

Las vistas de reconstrucción, tejido y diagnóstico se definieron como índices sobre la misma identidad de atlas. No fue necesario duplicar físicamente las representaciones para cada tarea.

## Poblaciones derivadas

La población de primer plano basada en Otsu contiene 1 750 221 coordenadas en las 152 WSI. La población de tejido de alta pureza contiene 666 807 coordenadas antes de aplicar límites específicos de cada tarea.

Para limitar poblaciones muy grandes sin quedarse únicamente con el inicio del barrido espacial, se usó selección sistemática centrada. Si una WSI contiene $N$ posiciones ordenadas y se desean $C<N$, se seleccionan los índices

$$
i_j=\left\lfloor\frac{(2j+1)N}{2C}\right\rfloor,
\qquad j=0,\ldots,C-1.
$$

Este muestreo reparte las posiciones sobre toda la secuencia espacial. Para reconstrucción se conservaron como máximo 3 000 posiciones de primer plano por WSI. Para tejido se conservaron como máximo 1 000 posiciones por combinación de WSI y clase. Para el diagnóstico de WSI se conservaron posteriormente las bolsas completas de primer plano, sin truncamiento.

# Preprocesamiento y corrupción de entrada

## Normalización

Un parche RGB `uint8` $\mathbf x^{(8)}\in\{0,\ldots,255\}^{3\times256\times256}$ se convirtió a `float32` y al dominio $[-1,1]$ mediante

$$
\mathbf x=2\frac{\mathbf x^{(8)}}{255}-1.
$$

La entrada y el objetivo limpio conservaron el orden NCHW. El decodificador también produjo valores en este dominio, pero su salida no fue limitada por `tanh`.

## Objetivo de eliminación de ruido

El VAE se entrenó como autoencodificador de eliminación de ruido. El objetivo siempre fue el parche limpio $\mathbf x$. La entrada fue

$$
\widetilde{\mathbf x}=
\begin{cases}
\mathcal C(\mathbf x), & u<0.3,\\
\mathbf x, & u\geq0.3,
\end{cases}
\qquad u\sim U(0,1),
$$

donde $\mathcal C$ es una perturbación de tinción HED más ruido gaussiano. Por tanto, aproximadamente el 30 % de las presentaciones recibieron corrupción y el resto permaneció limpio.

## Perturbación HED

Para aplicar la corrupción, el parche se llevó temporalmente a RGB en $[0,1]$ y se transformó al espacio de densidad óptica asociado a hematoxilina, eosina y residuo. Si $(h,e,r)$ denotan esos tres componentes, se aplicaron transformaciones afines independientes:

$$
\begin{aligned}
h'&=a_hh+b_h,\\
e'&=a_ee+b_e,\\
r'&=a_rr+b_r,
\end{aligned}
$$

con

$$
\begin{aligned}
a_h,a_e&\sim U(0.8,1.2),
&b_h,b_e&\sim U(-0.05,0.05),\\
a_r&\sim U(0.98,1.02),
&b_r&\sim U(-0.01,0.01).
\end{aligned}
$$

Los factores de hematoxilina y eosina permiten variaciones de intensidad más amplias que el residual. Después de aplicar la transformación inversa HED-RGB se añadió ruido

$$
\boldsymbol\eta\sim\mathcal N(\mathbf0,\sigma^2\mathbf I),
\qquad
\sigma\sim U(0,0.05).
$$

El resultado se recortó al dominio RGB válido y regresó a $[-1,1]$. La misma distribución de corrupción se utilizó en ambos VAE. La corrupción modifica únicamente la entrada; el objetivo de reconstrucción permanece limpio.

## Pseudocódigo conceptual

```python
def denoising_pair(rgb_uint8, rng):
    clean = rgb_uint8.float() / 255.0       # [3,256,256] en [0,1]

    if rng.uniform() < 0.30:
        h, e, r = rgb_to_hed(clean, eps=1e-6)
        h = rng.uniform(0.8, 1.2) * h + rng.uniform(-0.05, 0.05)
        e = rng.uniform(0.8, 1.2) * e + rng.uniform(-0.05, 0.05)
        r = rng.uniform(0.98, 1.02) * r + rng.uniform(-0.01, 0.01)
        corrupted = hed_to_rgb(h, e, r)
        std = rng.uniform(0.0, 0.05)
        corrupted = (corrupted + std * randn_like(corrupted)).clamp(0, 1)
    else:
        corrupted = clean

    return 2 * corrupted - 1, 2 * clean - 1
```

# Diseño común de los VAE

## Relación con ResNet-18

El punto de partida fue ResNet-18: cuatro etapas, dos bloques residuales básicos por etapa y conexiones cortas que suman una rama aprendida a una identidad o proyección. La arquitectura final no es una copia de ResNet-18, sino un VAE propio basado en esa organización.

Las modificaciones principales fueron:

1. reemplazar la cabeza de clasificación por un posterior espacial y un decodificador;
2. eliminar la reducción agresiva inicial para conservar detalle histológico;
3. usar un latente espacial de $32\times32$;
4. usar tres reducciones y tres aumentos de resolución;
5. sustituir BatchNorm y ReLU por operaciones con contraparte equivariante clara;
6. utilizar reducción antialias y aumento bilineal;
7. producir RGB crudo sin `tanh` final;
8. mantener la misma cantidad y ubicación macroscópica de bloques en las dos ramas.

Cada modelo contiene ocho bloques residuales en el codificador y ocho en el decodificador. En total contiene 43 convoluciones aprendidas, 40 normalizaciones y 34 activaciones con compuerta.

## Posterior gaussiano espacial

El codificador produce

$$
\boldsymbol\mu_\phi(\mathbf x),
\boldsymbol\ell_\phi(\mathbf x)
\in\mathbb R^{B\times16\times32\times32},
$$

donde $\boldsymbol\ell=\log\boldsymbol\sigma^2$. El posterior factoriza sobre canales y posiciones:

$$
q_\phi(\mathbf z\mid\mathbf x)
=\mathcal N\!\left(
\boldsymbol\mu_\phi(\mathbf x),
\operatorname{diag}[\exp(\boldsymbol\ell_\phi(\mathbf x))]
\right).
$$

Para evitar valores numéricamente extremos, se utiliza

$$
\overline{\boldsymbol\ell}
=\operatorname{clip}(\boldsymbol\ell,-8,4).
$$

El muestreo usa reparametrización:

$$
\boldsymbol\epsilon\sim\mathcal N(\mathbf0,\mathbf I),
\qquad
\mathbf z=\boldsymbol\mu+\exp(\overline{\boldsymbol\ell}/2)\odot\boldsymbol\epsilon.
$$

Durante entrenamiento se muestrea $\boldsymbol\epsilon$. Para reconstrucción, tareas supervisadas y geometría latente se utiliza $\mathbf z=\boldsymbol\mu$, salvo que un control indique explícitamente lo contrario.

## Bloque residual

Para entrada $\mathbf X$, un bloque calcula

$$
\mathbf Y=\Phi_2\!\left(\mathcal F(\mathbf X)+\mathcal S(\mathbf X)\right),
$$

donde $\mathcal F$ es la rama principal, $\mathcal S$ el camino corto y $\Phi_2$ una compuerta de salida. En un bloque de codificador:

$$
\mathcal F(\mathbf X)
=N_2\left(C_2\left(D\left(
\Phi_1(N_1(C_1(\mathbf X)))
\right)\right)\right),
$$

donde $D$ es identidad o reducción por dos. El camino corto aplica la misma reducción antes de una proyección cuando cambia la forma. En el decodificador, el aumento por dos ocurre antes de la primera convolución de cada rama.

Esta posición de las operaciones se mantiene en los dos modelos; solo cambian el tipo de convolución, normalización y compuerta.

## Remuestreo compartido

La reducción utiliza primero un filtro binomial separable fijo. Sea

$$
\mathbf b=\frac1{16}[1,4,6,4,1].
$$

El núcleo bidimensional es $\mathbf h=\mathbf b^\mathsf T\mathbf b$. Se aplica el mismo $\mathbf h$ de forma independiente a todos los canales, con paso 2. Al ser escalar e idéntico por componente, este operador también es válido para campos vectoriales.

El aumento utiliza interpolación bilineal por un factor de dos con la misma geometría en todos los canales, seguido por una convolución aprendida. No se usa PixelShuffle ni vecino más cercano.

# VAE convolucional de referencia

## Activación escalar aprendida

Para $\mathbf X\in\mathbb R^{B\times C\times H\times W}$, cada canal tiene parámetros $a_c,b_c$:

$$
\Phi_c(X_{bchw})
=X_{bchw}\,\sigma(a_cX_{bchw}+b_c),
\qquad
\sigma(t)=\frac{1}{1+e^{-t}}.
$$

Se inicializa $a_c=1$, $b_c=0$, de manera que inicialmente equivale a SiLU. Los parámetros pueden adaptar pendiente y desplazamiento durante el aprendizaje.

## Arquitectura

El VAE de referencia usa convoluciones ordinarias y GroupNorm con ocho grupos. El tallo es una convolución $7\times7$ de 3 a 32 canales. Las convoluciones restantes son $5\times5$.

| Etapa | Resolución | Canales | Operación principal |
|---|---:|---:|---|
| Entrada | 256 | 3 | RGB normalizado |
| Tallo | 256 | 32 | Conv $7\times7$ + norm + compuerta |
| Codificador A | 256 | 32 | 2 bloques residuales |
| Codificador B | 128 | 48 | transición + bloque |
| Codificador C | 64 | 64 | transición + bloque |
| Codificador D | 32 | 96 | transición + bloque |
| Posterior | 32 | 16 + 16 | cabezas independientes $\mu$ y `logvar` |
| Proyección latente | 32 | 96 | Conv + norm + compuerta |
| Decodificador D | 32 | 96 | 2 bloques residuales |
| Decodificador C | 64 | 64 | transición + bloque |
| Decodificador B | 128 | 48 | transición + bloque |
| Decodificador A | 256 | 32 | transición + bloque |
| Salida | 256 | 3 | Conv $5\times5$, sin activación final |

La última convolución RGB se inicializa en cero. Por ello, antes de la primera actualización la reconstrucción es cero en el dominio $[-1,1]$. El modelo contiene 3 958 435 parámetros aprendidos.

# Campos y capas equivariantes a $\mathrm{SO}(2)$

## Acción de una rotación

Sea

$$
R_\theta=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}.
$$

Un campo de frecuencia $n$ transforma como

$$
[T_\theta f_n](\mathbf u)
=\rho_n(\theta)f_n(R_{-\theta}\mathbf u),
$$

con

$$
\rho_0(\theta)=1,
\qquad
\rho_n(\theta)=
\begin{bmatrix}
\cos(n\theta)&-\sin(n\theta)\\
\sin(n\theta)&\cos(n\theta)
\end{bmatrix},\quad n\geq1.
$$

El experimento usa solamente:

- $F_0$: una componente escalar por copia;
- $F_1$: dos componentes por copia que giran como un vector plano.

Una disposición $c_0F_0+c_1F_1$ ocupa

$$
C_{\mathrm{físico}}=c_0+2c_1
$$

canales. Los $c_0$ escalares se almacenan primero; luego se guardan consecutivamente las dos componentes de cada una de las $c_1$ copias vectoriales.

## Restricción sobre el núcleo

Una convolución entre un campo de entrada $F_{\ell_i}$ y uno de salida $F_{\ell_o}$ es equivariante si su núcleo satisface

$$
K_{\ell_o\leftarrow\ell_i}(R_\theta\mathbf u)
=\rho_{\ell_o}(\theta)
K_{\ell_o\leftarrow\ell_i}(\mathbf u)
\rho_{\ell_i}(-\theta).
$$

En vez de aprender cada valor del núcleo libremente, se construye un banco fijo de funciones que satisfacen esta identidad y solo se aprenden sus coeficientes.

## Cuadrícula del núcleo

Para un soporte impar $k\times k$, con índices de fila y columna $i,j\in\{0,\ldots,k-1\}$:

$$
\begin{aligned}
x_{ij}&=j-\frac{k-1}{2},\\
y_{ij}&=\frac{k-1}{2}-i,\\
r_{ij}&=\sqrt{x_{ij}^2+y_{ij}^2},\\
\varphi_{ij}&=\operatorname{atan2}(y_{ij},x_{ij}).
\end{aligned}
$$

La inversión del eje vertical en $y_{ij}$ hace que las coordenadas cartesianas conserven orientación positiva aunque las filas de una imagen crezcan hacia abajo.

## Envolventes radiales

Para centro radial $\eta_s$ y anchura $\sigma_s$:

$$
g_s(r)=\exp\left[-\frac{(r-\eta_s)^2}{2\sigma_s^2}\right].
$$

Se usaron los perfiles:

| Soporte | Centros $\boldsymbol\eta$ | Anchuras $\boldsymbol\sigma$ |
|---|---|---|
| $7\times7$ | $(1,1.90395977,2.75)$ | $(0.3,0.3,0.3)$ |
| $9\times9$ | $(1,1.99907757,2.87711643,3.75)$ | $(0.3,0.3,0.3,0.3)$ |

El tallo usa $9\times9$; las demás convoluciones direccionables usan $7\times7$.

## Generadores por pareja de campos

Sea

$$
\mathbf J=\begin{bmatrix}0&-1\\1&0\end{bmatrix},\quad
\mathbf S=\begin{bmatrix}1&0\\0&-1\end{bmatrix},\quad
\mathbf T=\begin{bmatrix}0&1\\1&0\end{bmatrix}.
$$

Los generadores fueron:

$$
\begin{aligned}
\mathcal G_{0\leftarrow0}&=\{(0,[1])\},\\
\mathcal G_{1\leftarrow0}&=\{(1,\mathbf e_1),(1,\mathbf e_2)\},\\
\mathcal G_{0\leftarrow1}&=\{(1,\mathbf e_1^\mathsf T),(1,\mathbf e_2^\mathsf T)\},\\
\mathcal G_{1\leftarrow1}&=\{(0,\mathbf I),(0,\mathbf J),(2,\mathbf S),(2,\mathbf T)\}.
\end{aligned}
$$

Cada candidato de base se muestrea como

$$
\mathbf C_{s,q,A}[:,:,i,j]
=g_s(r_{ij})\,
\rho_{\ell_o}(\varphi_{ij})
A
\rho_{\ell_i}(-\varphi_{ij}).
$$

Los órdenes no constantes se fuerzan a cero en el centro $r=0$, donde la dirección angular no está definida. Los generadores de orden cero reciben además un impulso central exacto. En la arquitectura final se retienen órdenes espaciales hasta $q=2$.

## Ortonormalización del banco

Cada candidato $\mathbf C_m$ se vectoriza como una columna. Si

$$
\mathbf C\in
\mathbb R^{(d_od_ik^2)\times M},
$$

se calcula una QR reducida

$$
\mathbf C=\mathbf Q\mathbf R.
$$

Para fijar la ambigüedad de signo, cada columna de $\mathbf Q$ se orienta de modo que su entrada de mayor valor absoluto sea positiva; los empates se resuelven por el primer índice. El banco usado por la capa es

$$
\mathbf B=\sqrt{d_o}\,\mathbf Q^\mathsf T
\in\mathbb R^{M\times(d_od_ik^2)}.
$$

Las filas de $\mathbf B$ permanecen fijas. Solo los coeficientes de expansión reciben gradiente.

## Expansión tensorial del núcleo

Considérese una pareja con $c_i$ copias de entrada, $c_o$ copias de salida, dimensiones internas $d_i,d_o$ y $M$ bases. Para cada pareja de copias se aprende

$$
\boldsymbol\alpha_{ba}\in\mathbb R^M.
$$

Al apilar los coeficientes:

$$
\mathbf A\in\mathbb R^{(c_oc_i)\times M}.
$$

La expansión completa es una multiplicación matricial:

$$
\mathbf G=\mathbf A\mathbf B
\in\mathbb R^{(c_oc_i)\times(d_od_ik^2)}.
$$

Se reorganiza sin cambiar el orden lógico:

$$
\begin{aligned}
&[c_o,c_i,d_o,d_i,k,k]\\
&\quad\xrightarrow{\operatorname{permute}(0,2,1,3,4,5)}
[c_o,d_o,c_i,d_i,k,k]\\
&\quad\xrightarrow{\operatorname{reshape}}
[c_od_o,c_id_i,k,k].
\end{aligned}
$$

Por índices:

$$
K[bd_o+r_o,ad_i+r_i,i,j]
=\sum_{m=0}^{M-1}
\alpha[b,a,m]B[m,r_o,r_i,i,j].
$$

Esta ecuación es suficiente para implementar el bloque sin conocer teoría de representaciones. Las bases determinan cómo se relacionan componentes y posiciones; $\alpha$ determina cuánto participa cada patrón.

Para disposiciones mixtas se expanden cuatro bloques:

$$
\mathbf K=
\begin{bmatrix}
\mathbf K^{0\leftarrow0}&\mathbf K^{0\leftarrow1}\\
\mathbf K^{1\leftarrow0}&\mathbf K^{1\leftarrow1}
\end{bmatrix}.
$$

Después de ensamblar el núcleo denso, la entrada se procesa con una sola `conv2d`. La equivarianza está en la parametrización del peso, no en un bucle que rote filtros durante cada inferencia.

Los tamaños de banco fueron:

| Soporte | $0\leftarrow0$ | $1\leftarrow0$ | $0\leftarrow1$ | $1\leftarrow1$ |
|---|---:|---:|---:|---:|
| $7\times7$ | 4 | 6 | 6 | 14 |
| $9\times9$ | 5 | 8 | 8 | 18 |

## Contracción batched de los cuatro bloques

En capas ocultas con igual número de copias $F_0$ y $F_1$, los cuatro productos se pueden agrupar. Se rellenan los coeficientes hasta

$$
\widetilde{\mathbf A}
\in\mathbb R^{4\times(c_oc_i)\times14}
$$

y las bases hasta

$$
\widetilde{\mathbf B}
\in\mathbb R^{4\times14\times196}.
$$

Entonces

$$
\widetilde{\mathbf G}
=\operatorname{bmm}(\widetilde{\mathbf A},\widetilde{\mathbf B}).
$$

El relleno solo uniforma dimensiones. Después de la multiplicación se conservan las entradas correspondientes a cada forma física. Este agrupamiento no cambia la ecuación del núcleo.

## Pseudocódigo de la convolución

```python
def expand_pair(coeff, basis, co, ci, do, di, k):
    # coeff: [(co*ci), M]
    # basis: [M, (do*di*k*k)]
    flat = coeff @ basis
    block = flat.reshape(co, ci, do, di, k, k)
    block = block.permute(0, 2, 1, 3, 4, 5)
    return block.reshape(co * do, ci * di, k, k)


def assemble_f01(k00, k01, k10, k11):
    # filas: F0 de salida, F1 de salida
    # columnas: F0 de entrada, F1 de entrada
    top = torch.cat([k00, k01], dim=1)
    bottom = torch.cat([k10, k11], dim=1)
    return torch.cat([top, bottom], dim=0)


def forward_f01(x, coefficient_sets, fixed_bases, layouts, k):
    blocks = {}
    for pair in ("00", "01", "10", "11"):
        blocks[pair] = expand_pair(
            coefficient_sets[pair], fixed_bases[pair],
            layouts[pair].co, layouts[pair].ci,
            layouts[pair].do, layouts[pair].di, k,
        )
    kernel = assemble_f01(
        blocks["00"], blocks["01"], blocks["10"], blocks["11"]
    )
    return torch.nn.functional.conv2d(x, kernel, padding=k // 2)
```

## Normalización por tipos de campo

Para escalares $F_0$ se usa una normalización por grupos. Si existen $c_0$ copias, se dividen en ocho grupos y se calculan media y varianza sobre copias del grupo y posiciones espaciales. Cada copia conserva una escala $\gamma_c$ y un sesgo $\beta_c$.

Para $F_1$ no se puede restar una media vectorial arbitraria ni normalizar componentes por separado. Se agrupan las copias en cuatro grupos y se calcula

$$
s_g=sqrt{
\operatorname{media}_{c\in g,r\in\{1,2\},h,w}
X_{c,r,h,w}^2+10^{-5}
}.
$$

Luego

$$
\widehat{\mathbf X}_{c,:,h,w}
=\gamma_c\frac{\mathbf X_{c,:,h,w}}{s_g}.
$$

No se añade un sesgo vectorial, porque un vector constante no nulo seleccionaría una orientación privilegiada.

## Compuerta radial

Los campos $F_0$ usan la misma compuerta escalar del VAE de referencia. Para una copia vectorial

$$
\mathbf v=(v_1,v_2)^\mathsf T,
$$

se calcula

$$
r=\sqrt{v_1^2+v_2^2+10^{-4}},
$$

y se aplica un único escalar a ambas componentes:

$$
\Phi(\mathbf v)
=\mathbf v\,\sigma(ar+b).
$$

Como $\|R_\theta\mathbf v\|_2=\|\mathbf v\|_2$,

$$
\Phi(R_\theta\mathbf v)=R_\theta\Phi(\mathbf v).
$$

Una activación componente a componente, como ReLU independiente sobre $v_1$ y $v_2$, no cumple en general esta identidad.

# Arquitectura completa del VAE-$\mathrm{SO}(2)$

Las disposiciones ocultas fueron:

| Nombre | Disposición | Canales físicos | Resolución en codificador |
|---|---|---:|---:|
| $R$ | $3F_0$ | 3 | 256 |
| $A$ | $16F_0+16F_1$ | 48 | 256 |
| $B$ | $24F_0+24F_1$ | 72 | 128 |
| $C$ | $32F_0+32F_1$ | 96 | 64 |
| $D$ | $48F_0+48F_1$ | 144 | 32 |
| $L$ | $16F_0$ | 16 | 32 |

El recorrido es:

1. tallo $R\rightarrow A$ con soporte $9\times9$;
2. dos bloques $A\rightarrow A$;
3. transición con reducción $A\rightarrow B$ y un bloque $B\rightarrow B$;
4. transición $B\rightarrow C$ y un bloque $C\rightarrow C$;
5. transición $C\rightarrow D$ y un bloque $D\rightarrow D$;
6. cabezas independientes $D\rightarrow L$ para $\mu$ y `logvar`;
7. proyección $L\rightarrow D$;
8. recorrido simétrico $D\rightarrow C\rightarrow B\rightarrow A$ con tres aumentos;
9. proyección $A\rightarrow R$ sin activación final.

El posterior se mantiene enteramente en $F_0$. Así, el muestreo gaussiano se realiza componente a componente con la misma semántica que en el VAE convencional.

La inicialización de un conjunto de coeficientes usa

$$
\alpha_{ba,m}\sim\mathcal N(0,\sigma_\alpha^2),
\qquad
\sigma_\alpha=\frac{1}{\sqrt{n_fc_iM}},
$$

con $n_f=1$ en la entrada escalar y $n_f=2$ en capas mixtas. La cabeza RGB final se inicializa en cero, igual que en la rama convencional.

El VAE-$\mathrm{SO}(2)$ contiene 1 180 035 parámetros aprendidos. Tiene menos parámetros libres que el VAE de referencia, pero mayor costo de convolución debido a soportes más grandes y más canales físicos. La comparación controla los datos y el protocolo, no iguala simultáneamente parámetros y operaciones.

# Pérdida y entrenamiento de los VAE

## Pérdida de reconstrucción

La pérdida total fue

$$
\mathcal L
=\mathcal L_{\mathrm{MAE}}
+0.1\mathcal L_{\mathrm{SSIM}}
+\beta\mathcal L_{\mathrm{KL}}.
$$

Para salida cruda $\widehat{\mathbf x}$ y objetivo limpio $\mathbf x$ en $[-1,1]$:

$$
\mathcal L_{\mathrm{MAE}}
=\operatorname{media}|\widehat{\mathbf x}-\mathbf x|.
$$

SSIM se calcula en el dominio de imagen $[0,1]$. La salida se proyecta únicamente para este término:

$$
\widehat{\mathbf x}_{01}
=\operatorname{clip}\left(\frac{\widehat{\mathbf x}+1}{2},0,1\right),
\qquad
\mathbf x_{01}=\frac{\mathbf x+1}{2}.
$$

Para ventanas locales $a,b$:

$$
\operatorname{SSIM}(a,b)
=\frac{(2\mu_a\mu_b+C_1)(2\sigma_{ab}+C_2)}
{(\mu_a^2+\mu_b^2+C_1)(\sigma_a^2+\sigma_b^2+C_2)},
$$

con $C_1=0.01^2$, $C_2=0.03^2$, ventana gaussiana $11\times11$ y desviación 1.5. El término usado por la pérdida es

$$
\mathcal L_{\mathrm{SSIM}}=1-\operatorname{media}(\operatorname{SSIM}).
$$

MAE continúa penalizando valores fuera del intervalo aunque el recorte de SSIM tenga gradiente nulo allí.

## Regularización KL

Para cada elemento del posterior:

$$
D_{\mathrm{KL}}
\left(\mathcal N(\mu,\sigma^2)\|\mathcal N(0,1)\right)
=-\frac12(1+\log\sigma^2-\mu^2-\sigma^2).
$$

Con $\boldsymbol\ell=\log\boldsymbol\sigma^2$:

$$
\mathcal L_{\mathrm{KL}}
=\operatorname{media}_{b,c,h,w}
\left[-\frac12
(1+\overline\ell_{bchw}
-\mu_{bchw}^2
-e^{\overline\ell_{bchw}})
\right].
$$

El peso alcanza $\beta_{\max}=0.01$. Se inicia en cero y aumenta linealmente durante las primeras 6 000 actualizaciones; luego permanece constante.

## Optimización

Ambos modelos se entrenaron durante 60 000 actualizaciones confirmadas. Se utilizó AdamW con:

| Parámetro | Valor |
|---|---:|
| tasa máxima efectiva | $10^{-3}$ |
| $\beta_1,\beta_2$ | $(0.9,0.999)$ |
| $\epsilon$ | $10^{-8}$ |
| decaimiento de pesos | $10^{-5}$ |
| recorte de norma global | $1.0$ |
| calentamiento de tasa | 600 actualizaciones |
| tasa mínima final | $10^{-5}$ |
| calendario posterior | coseno, sin reinicios |

Las compuertas se entrenaron con la mitad del multiplicador de tasa y sin decaimiento. Sesgos y parámetros de normalización tampoco recibieron decaimiento. La validación incluyó una vista limpia y una vista de eliminación de ruido determinista.

Al finalizar se seleccionó el estado de la actualización 60 000 para ambos modelos y se congelaron codificador y decodificador. Las etiquetas posteriores no modificaron estos pesos.

# Extracción de representaciones congeladas

Para cada parche limpio seleccionado se calculó

$$
\boldsymbol\mu_i=E_\mu(\mathbf x_i)
\in\mathbb R^{16\times32\times32}.
$$

Los dos codificadores recibieron exactamente la misma secuencia de parches. Las representaciones se almacenaron en `float32`, ordenadas y enlazadas con la identidad del atlas. Esta decisión preserva toda la estructura espacial del posterior; no se aplicó promedio, cuantización ni reducción de dimensión antes de las tareas posteriores.

El lector de una reimplementación debe garantizar:

1. correspondencia exacta entre registro y $(\text{wsi\_id},x,y)$;
2. misma población y mismo orden para ambas ramas;
3. forma `[16,32,32]` y tipo `float32`;
4. ausencia de mezcla entre entrenamiento, validación y prueba;
5. conservación de la bolsa completa cuando la tarea es diagnóstico de WSI.

# Tareas supervisadas posteriores

## Clasificación de tejido a resolución de parche

La tarea predice tumor, estroma o necrosis para cada celda elegible. Al devolver las predicciones a $(x,y)$ se obtiene una segmentación dispersa de la WSI a resolución de parche. Las posiciones no elegibles permanecen sin predicción; no se asignan a una clase de fondo.

El clasificador recibe $\boldsymbol\mu\in\mathbb R^{16\times32\times32}$ y usa tres etapas convolucionales

$$
16\rightarrow32\rightarrow64\rightarrow128,
$$

con reducción espacial, GroupNorm y GELU. Un promedio adaptativo produce un vector de 128 componentes y una capa lineal genera tres logits.

Se entrenaron clasificadores independientes con 250, 500, 1 000, 2 500 y 5 671 ejemplos por clase. Los subconjuntos fueron:

- balanceados entre clases;
- anidados, de modo que cada presupuesto contiene al anterior;
- estratificados por WSI;
- idénticos para las dos representaciones.

Cada par de clasificadores comenzó con los mismos pesos y recibió los mismos índices por época. La selección se realizó por macro-F1 de validación. Se permitió un máximo de 30 épocas y la parada temprana solo se activó después de diez épocas completas.

## Clasificación diagnóstica a nivel de WSI

Cada WSI es una bolsa

$$
\mathcal B_w
=\{(\boldsymbol\mu_i,\mathbf g_i)\}_{i=1}^{N_w},
\qquad
\mathbf g_i=(x_i/256,y_i/256)\in\mathbb Z^2.
$$

El tamaño $N_w$ varía entre láminas. El clasificador debe producir una sola etiqueta diagnóstica sin ordenar artificialmente la importancia de las instancias ni construir atención global parche a parche de costo $O(N_w^2)$.

### Codificador de cada parche

Tres convoluciones con paso dos y anchos

$$
16\rightarrow64\rightarrow128\rightarrow192
$$

seguidas por promedio espacial producen

$$
\mathbf X_0\in\mathbb R^{N\times192}.
$$

Cada fila conserva la identidad de un parche.

### Grafo local

Se usa $D=192$, $H=6$, $d_h=32$ y un máximo de $L=25$ vecinos. Para cada posición:

$$
\mathcal N(i)=
\{j:\|\mathbf g_i-\mathbf g_j\|_\infty\leq2\}.
$$

Es una ventana espacial de hasta $5\times5$ que conserva huecos donde no hay tejido.

En cada uno de dos bloques locales independientes:

$$
\mathbf P=\operatorname{LN}(\mathbf X),
\qquad
[\mathbf Q|\mathbf K|\mathbf V]
=\mathbf P\mathbf W_{qkv}.
$$

Después de reorganizar:

$$
\mathbf Q,\mathbf K,\mathbf V
\in\mathbb R^{N\times H\times d_h}.
$$

Un tensor de índices reúne para cada consulta las claves y valores de sus vecinos:

$$
\widetilde{\mathbf K},\widetilde{\mathbf V}
\in\mathbb R^{N\times H\times(L+1)\times d_h}.
$$

La ranura adicional es una clave nula aprendida con valor cero. Las puntuaciones son

$$
S_{ihr}
=\frac{\langle\mathbf Q_{ih:},
\widetilde{\mathbf K}_{ihr:}\rangle}{\sqrt{d_h}}
+B_{ihr},
$$

donde $B$ contiene un sesgo aprendido para la distancia cuadrática entre parches y $-\infty$ en posiciones inválidas. Los seis códigos de distancia son $\{0,1,2,4,5,8\}$.

$$
A_{ihr}=\operatorname{softmax}_r(S_{ihr}),
\qquad
Z_{iha}=\sum_r A_{ihr}\widetilde V_{ihra}.
$$

Las contracciones corresponden a:

```python
score = torch.einsum("nhd,nhld->nhl", q, neighbor_keys)
context = torch.einsum("nhl,nhld->nhd", weight, neighbor_values)
```

Después se aplican proyección, residual y una SwiGLU pre-normalizada. Dos bloques producen

$$
\mathbf X_2\in\mathbb R^{N\times192}.
$$

### Resumen global mediante sigmoides

Se crean un token CLS y 16 tokens REG:

$$
\mathbf T_0
=[\mathrm{CLS};\mathrm{REG}_1;\ldots;\mathrm{REG}_{16}]
\in\mathbb R^{17\times192}.
$$

Estos 17 tokens consultan las $N$ representaciones locales. Para consulta $t$, cabeza $h$ y parche $j$:

$$
L_{thj}
=\frac{\langle\mathbf Q_{th:},\mathbf K_{jh:}\rangle}{\sqrt{d_h}}
+b_h-\log N,
$$

$$
A_{thj}=\sigma(L_{thj}),
\qquad
\mathbf C_{th:}=\sum_{j=1}^{N}A_{thj}\mathbf V_{jh:}.
$$

La sigmoide permite que múltiples parches reciban peso alto simultáneamente. La corrección $-\log N$ estabiliza la magnitud inicial: si los demás términos son cero,

$$
\sum_j\sigma(-\log N)=\frac{N}{N+1}\approx1.
$$

### Lectura final CLS

CLS consulta los 17 resúmenes con atención softmax convencional. Solo su fila actualizada pasa por normalización y una capa lineal

$$
\mathbb R^{192}\rightarrow\mathbb R^5
$$

que produce logits en el orden CC, EC, HGSC, LGSC y MC.

La conectividad total crece linealmente con $N$:

$$
2(25N)+17N+17,
$$

en lugar de $N^2$. El modelo se entrenó con bolsas completas y lote de una WSI. La selección se hizo con macro-F1 de validación y entropía cruzada como desempate.

# Visualizaciones y sondeos del espacio latente

Las visualizaciones se trataron como experimentos. Cada una define explícitamente el tensor de entrada, la transformación, la escala gráfica, la cantidad numérica asociada y los límites de interpretación.

## Tableros de entrenamiento y tareas supervisadas

El seguimiento del entrenamiento se diseñó para responder dos preguntas distintas: si la optimización continuaba progresando y si ese progreso se transfería a datos limpios no usados para actualizar los pesos. Las curvas de entrenamiento se agregaron entre procesos para cada contador de actualización del optimizador; las métricas de validación se calcularon como medias ponderadas por el número declarado de observaciones. Por ello, no se mezclaron promedios de lotes de tamaños diferentes como si cada lote tuviera el mismo peso.

Las series de entrenamiento se mostraron tanto en su forma cruda como mediante una media móvil centrada de 501 observaciones. Si $y_t$ es el valor crudo en la actualización $t$, la curva suavizada es

$$
\widetilde y_t
=\frac{1}{|\mathcal W_t|}
\sum_{s\in\mathcal W_t}y_s,
\qquad
\mathcal W_t
=\{s:|s-t|\leq250\},
$$

recortando $\mathcal W_t$ en los extremos de la serie. Este suavizado es exclusivamente gráfico: no interviene en selección de modelos, pruebas estadísticas ni cálculos posteriores, y nunca reemplaza los valores crudos almacenados.

El tablero principal incluyó seis paneles sincronizados por actualización:

1. objetivo total de entrenamiento suavizado y objetivo de validación sobre entradas limpias;
2. término L1 de entrenamiento suavizado y L1 de validación limpia;
3. término $1-\operatorname{SSIM}$ de entrenamiento suavizado y su contraparte de validación;
4. PSNR de reconstrucción en un conjunto fijo de 25 parches limpios, evaluado en cada frontera archivada;
5. razón de equivarianza latente en esos mismos 25 parches, promediada sobre $90^\circ$, $180^\circ$ y $270^\circ$ y mostrada con eje vertical logarítmico;
6. tasa de aprendizaje efectiva.

Los tres primeros paneles separan progreso de optimización y generalización limpia; el cuarto vuelve interpretable la magnitud del error en unidades de imagen; el quinto permite detectar desviaciones multiplicativas que serían difíciles de leer en escala lineal; y el sexto permite relacionar cambios de pendiente con el programa de optimización. Los 25 parches fijos constituyen un sondeo pareado y reproducible, no una estimación de rendimiento poblacional.

Para las métricas de reconstrucción de ese sondeo se conservaron los valores por imagen. Las cajas resumen mediana y cuartiles, mientras que los puntos crudos y las líneas pareadas muestran cada observación antes y después, o para ambos modelos, sin ocultarla bajo el resumen. Por imagen se calcularon MAE y MSE en el dominio normalizado, y PSNR y SSIM en la imagen proyectada a $[0,1]$. Cada panel informa media, desviación estándar poblacional y $n$, pero la inferencia formal agrupa por WSI cuando corresponde para no tratar parches correlacionados como réplicas independientes.

En las tareas de clasificación, la matriz de confusión cruda

$$
\mathbf C=(C_{ij}),
$$

donde $C_{ij}$ cuenta ejemplos de la clase real $i$ predichos como $j$, se acompañó de su normalización por fila:

$$
P_{ij}
=\frac{C_{ij}}{\sum_j C_{ij}}.
$$

Cada celda muestra el conteo y el porcentaje dentro de su clase real. Esta forma permite distinguir una clase numerosa de una clase sistemáticamente difícil. Como complemento, para cada clase $c$ se reporta

$$
F1_c
=\frac{2\,TP_c}{2\,TP_c+FP_c+FN_c},
$$

y la macro-F1 es la media no ponderada de los $F1_c$. Así, todas las clases tienen el mismo peso aunque sus prevalencias difieran.

La eficiencia de etiquetas se visualizó entrenando el mismo clasificador de tejido con presupuestos

$$
\mathcal B=\{250,500,1000,2500,5671\}
$$

y graficando macro-F1 contra $x_i=\log_{10}(b_i)$. El eje logarítmico representa que el cambio relevante es multiplicativo en el número de etiquetas. La curva se resume con el área normalizada bajo la curva de aprendizaje

$$
\operatorname{AULC}
=\frac{1}{x_5-x_1}
\sum_{i=1}^{4}
\frac{F_i+F_{i+1}}{2}(x_{i+1}-x_i),
$$

donde $F_i$ es la macro-F1 sin suavizar al presupuesto $b_i$. La AULC responde si una representación es útil a lo largo de todo el régimen de datos, no solo en el presupuesto máximo. Las curvas de desarrollo se usaron para selección; el conjunto sellado de prueba se reservó para la evaluación final y nunca se inspeccionó para escoger presupuestos, hiperparámetros o variantes.

## Reconstrucciones deterministas

Para un parche limpio:

$$
\boldsymbol\mu=E_\mu(\mathbf x),
\qquad
\widehat{\mathbf x}=D(\boldsymbol\mu).
$$

No se muestrea $\boldsymbol\epsilon$. Las rejillas cualitativas mantienen el mismo parche por columna para los dos modelos. La imagen mostrada es

$$
\widehat{\mathbf x}_{\mathrm{vis}}
=\operatorname{clip}
\left(\frac{\widehat{\mathbf x}+1}{2},0,1\right).
$$

El recorte solo pertenece a la visualización y a métricas de imagen. MAE y MSE principales usan la salida cruda en $[-1,1]$:

$$
\operatorname{MSE}
=\operatorname{media}(\widehat{\mathbf x}-\mathbf x)^2.
$$

PSNR usa el dominio $[0,1]$:

$$
\operatorname{PSNR}
=10\log_{10}\frac{1}{\operatorname{MSE}_{[0,1]}}.
$$

Se produjeron distribuciones por parche y comparaciones pareadas por WSI. Las cajas de parches describen dispersión computacional, mientras los intervalos de comparación se calculan agrupando por WSI.

## PCA espacial con escala común por modelo

Sea

$$
\boldsymbol\mu_{m,n}
\in\mathbb R^{16\times32\times32}
$$

la media posterior del modelo $m$ y parche $n$. Para una posición $p$ del disco central se define el descriptor

$$
\mathbf f_{m,n}(p)
=(\mu_{1}(p),\ldots,\mu_{16}(p))
\in\mathbb R^{16}.
$$

Se apilan los descriptores de los 25 parches y 616 posiciones centrales:

$$
\mathbf F_m
\in\mathbb R^{(25\cdot616)\times16}.
$$

Después de centrar por la media global $\overline{\mathbf f}_m$, se obtienen los tres vectores principales

$$
\mathbf W_m\in\mathbb R^{16\times3}.
$$

La proyección en una posición es

$$
\mathbf y_{m,n}(p)
=(\mathbf f_{m,n}(p)-\overline{\mathbf f}_m)\mathbf W_m.
$$

Cada vector principal se orienta haciendo positiva su carga de mayor valor absoluto. Para conservar escalas relativas entre las tres componentes se usa un único valor por modelo:

$$
s_m=Q_{0.99}
\left(\{|y_{m,n,k}(p)|\}_{n,p,k}\right).
$$

Los canales mostrados son

$$
\operatorname{RGB}_k
=\operatorname{clip}\left(\frac12+\frac{y_k}{2s_m},0,1\right).
$$

La misma base y escala se usan en los 25 parches de un modelo. Sin embargo, cada modelo tiene su propia PCA; por ello, un color no representa la misma dirección latente entre modelos.

### Rugosidad espacial relativa

Sea $\mathcal E$ el conjunto de pares horizontales y verticales vecinos dentro del disco central $\mathcal D$. Para cada parche:

$$
r_{\mathrm{arista}}
=\frac{
\sqrt{\frac1{16|\mathcal E|}
\sum_c\sum_{(p,q)\in\mathcal E}
[\mu_c(p)-\mu_c(q)]^2}}
{
\sqrt{\frac1{16|\mathcal D|}
\sum_c\sum_{p\in\mathcal D}
[\mu_c(p)-\overline\mu_c]^2}}
.
$$

El numerador mide variación local y el denominador la variación espacial total del mismo campo. Un valor menor indica menos cambio entre vecinos en relación con la variación global, pero también podría surgir de suavizado o colapso. No es una medida general de calidad ni equivarianza.

## PCA por parche con estilo de mapa cromático

Como vista puramente visual, para cada parche y modelo se reorganiza el mapa completo como

$$
\mathbf X_{m,n}\in\mathbb R^{1024\times16}.
$$

Se ajusta una PCA independiente y se obtienen

$$
\mathbf Z_{m,n}
=(\mathbf X_{m,n}-\mathbf1\overline{\mathbf x}_{m,n}^\mathsf T)
\mathbf V_{m,n}
\in\mathbb R^{1024\times3}.
$$

Las tres componentes usan un mínimo y máximo conjunto del mismo mapa:

$$
\operatorname{RGB}_{m,n}
=\frac{\mathbf Z_{m,n}-\min\mathbf Z_{m,n}}
{\max\mathbf Z_{m,n}-\min\mathbf Z_{m,n}}.
$$

El campo $32\times32$ se aumenta bilinealmente solo para mostrarlo. Esta vista maximiza contraste interno, pero sus colores y magnitudes no son comparables entre parches ni modelos.

## Sondeo lineal de apariencia RGB

La imagen original se reduce a

$$
\mathbf T_n\in[0,1]^{3\times32\times32}
$$

mediante interpolación bilineal antialias. Para cada modelo se ajusta una transformación afín común a todas las posiciones:

$$
\widehat{\mathbf t}_{m,n}(p)
=\mathbf W_m\mathbf f_{m,n}(p)+\mathbf b_m,
\qquad
\mathbf W_m\in\mathbb R^{3\times16}.
$$

Las variables se estandarizan usando solo 20 parches de ajuste. Se resuelve una regresión ridge con $\lambda=10^{-3}$, sin penalizar el intercepto. Los cinco parches restantes no participan en el ajuste.

El puntaje de un parche de evaluación es

$$
R^2_{m,n}
=1-
\frac{\sum_{p,c}[\widehat t_{m,n,c}(p)-T_{n,c}(p)]^2}
{\sum_{p,c}[\overline T_{\mathrm{ajuste},c}-T_{n,c}(p)]^2}.
$$

La predicción se recorta a $[0,1]$ únicamente para mostrarla. El sondeo pregunta si un descriptor local de 16 dimensiones conserva apariencia cromática recuperable mediante una lectura lineal sin usar vecinos. No es una reconstrucción del decodificador ni una medida de equivarianza.

## Órbitas densas de rotación

Para cada uno de 25 parches fijos se generan vistas

$$
\mathbf x_\theta=R_\theta\mathbf x,
\qquad
\theta\in\{0^\circ,1^\circ,\ldots,359^\circ\}.
$$

La orientación del operador se valida previamente con patrones asimétricos. Las rotaciones no cardinales usan interpolación bilineal; $90^\circ$, $180^\circ$ y $270^\circ$ usan permutaciones exactas de la cuadrícula.

Cada vista produce

$$
\mathbf z_\theta=E_\mu(\mathbf x_\theta).
$$

Para reducir efectos de borde, se restringe el mapa latente a un disco central de radio 14, con 616 posiciones y 9 856 valores. Los 360 vectores aplanados forman

$$
\mathbf Z\in\mathbb R^{360\times9856}.
$$

Se ajusta una PCA independiente por parche y modelo. Las dos primeras coordenadas se usan para visualizar la órbita. Esa PCA solo representa la trayectoria; no establece que la órbita sea realmente bidimensional ni permite comparar escalas entre paneles.

### Diferencias cíclicas

Para paso angular $\delta$:

$$
\Delta\mathbf z_\theta
=\mathbf z_{\theta+\delta}-\mathbf z_\theta,
$$

$$
\Delta^2\mathbf z_\theta
=\Delta\mathbf z_{\theta+\delta}-\Delta\mathbf z_\theta.
$$

La regularidad local se resume mediante

$$
r_{\mathrm{lin}}(\mathbf z)
=\frac{
\sqrt{\operatorname{media}_{\theta,d}
(\Delta^2z_{\theta d})^2}}
{
\sqrt{\operatorname{media}_{\theta,d}
(\Delta z_{\theta d})^2}+\epsilon}.
$$

La misma cantidad se calcula en la órbita RGB y se define

$$
\widetilde r_{\mathrm{lin}}(\mathbf z)
=\frac{r_{\mathrm{lin}}(\mathbf z)}
{r_{\mathrm{lin}}(\mathbf x)+\epsilon}.
$$

Esta división controla parcialmente la irregularidad introducida por interpolar la entrada. Se evalúan $\delta=1^\circ$ y $5^\circ$. También se estudian longitud de trayectoria, dispersión de tamaños de paso, curvatura discreta, espectro angular y dimensiones locales de los vectores tangentes.

## Controles exactos de $C_4$

Las rotaciones cardinales forman

$$
C_4=\{e,r,r^2,r^3\},
\qquad r=R_{90^\circ}.
$$

Para cada parche:

$$
\mathbf e_k=E_\mu(r^k\mathbf x),
\qquad
\mathbf a_k=\rho(r)^kE_\mu(\mathbf x).
$$

$\mathbf e_k$ es el ciclo codificado y $\mathbf a_k$ el ciclo obtenido aplicando la acción prescrita al latente original. La equivarianza estricta del codificador exigiría $\mathbf e_k=\mathbf a_k$. La consistencia del decodificador puede ser mejor aun cuando esos tensores no coincidan.

## Acción, canonización y composición extremo a extremo

Defina

$$
\mathbf z_0=E_\mu(\mathbf x),
\qquad
\mathbf y_0=D(\mathbf z_0),
$$

$$
\mathbf z_\theta=E_\mu(R_\theta\mathbf x),
\qquad
\mathbf y_{\mathrm{in}}=D(\mathbf z_\theta).
$$

Se comparan tres rutas:

1. **acción:** $D(R_\theta\mathbf z_0)$ frente a $R_\theta\mathbf y_0$;
2. **canonización:** $D(R_{-\theta}\mathbf z_\theta)$ frente a $\mathbf y_0$;
3. **extremo a extremo:** $D(\mathbf z_\theta)$ frente a $R_\theta\mathbf y_0$.

Para un conjunto de ángulos $\Omega$ y máscara espacial $M$:

$$
\|\mathbf a\|_{\Omega,M}
=\left[
\frac1{|\Omega||M|C}
\sum_{\theta\in\Omega}
\sum_{u\in M}
\sum_{c=1}^{C}a_{\theta cu}^2
\right]^{1/2}.
$$

Las razones son

$$
r_{\mathrm{act}}
=\frac{
\|D(R_\theta\mathbf z_0)-R_\theta\mathbf y_0\|_{\Omega,M}}
{\|\mathbf y_0-R_\theta\mathbf y_0\|_{\Omega,M}+10^{-8}},
$$

$$
r_{\mathrm{can}}
=\frac{
\|D(R_{-\theta}\mathbf z_\theta)-\mathbf y_0\|_{\Omega,M}}
{\|D(\mathbf z_\theta)-\mathbf y_0\|_{\Omega,M}+10^{-8}},
$$

$$
r_{\mathrm{e2e}}
=\frac{
\|D(\mathbf z_\theta)-R_\theta\mathbf y_0\|_{\Omega,M}}
{\|\mathbf y_0-R_\theta\mathbf y_0\|_{\Omega,M}+10^{-8}}.
$$

El denominador representa no aplicar la intervención candidata. Una razón igual a uno no mejora esa referencia; una razón menor indica reducción del error. Estas razones se acompañan con RMS y MAE absolutos, SSIM, valores fuera de rango y errores de gradiente para evitar conclusiones basadas únicamente en un cociente pequeño.

## Diagnósticos armónicos e internos

Sobre las órbitas se analizan:

- energía por frecuencia angular mediante transformada discreta de Fourier;
- concentración en armónicos dominantes;
- consistencia entre tangentes separados angularmente;
- respuesta de las 48 copias $F_1$ de la etapa profunda;
- posibilidad de aproximar una acción compartida de baja dimensión;
- separación descriptiva entre contenido y pose.

Estos son sondeos y no forman parte del entrenamiento. Un patrón circular en PCA o una frecuencia dominante no demuestra por sí solo una acción global, una variedad de baja dimensión o una factorización contenido-pose.

# Geometría funcional del decodificador

## Motivación

Dos vectores latentes alejados pueden decodificar imágenes parecidas. Por tanto, la distancia euclidiana en el latente no necesariamente refleja el cambio funcional del modelo. Se estudia la geometría inducida por el decodificador congelado

$$
D:\mathcal Z\rightarrow\mathcal X.
$$

En un punto $\mathbf z$, el Jacobiano $J_D(\mathbf z)$ transforma una perturbación latente $\mathbf u$ en un cambio de imagen de primer orden. La forma pullback es

$$
g_{\mathbf z}(\mathbf u,\mathbf v)
=\langle J_D(\mathbf z)\mathbf u,
J_D(\mathbf z)\mathbf v\rangle,
$$

y el operador asociado

$$
G(\mathbf z)=J_D(\mathbf z)^\mathsf TJ_D(\mathbf z).
$$

Es semidefinido positivo. Una dirección en el núcleo instantáneo de $J_D$ tiene longitud funcional cero de primer orden, pero eso no prueba la existencia de una fibra conectada.

## Productos sin materializar el Jacobiano

El latente tiene dimensión

$$
16\cdot32\cdot32=16\,384.
$$

No se construye un Jacobiano denso. Para vector $\mathbf v$ se calcula

$$
\mathbf w=J_D(\mathbf z)\mathbf v
$$

mediante un producto Jacobiano-vector (JVP), seguido por

$$
G(\mathbf z)\mathbf v
=J_D(\mathbf z)^\mathsf T\mathbf w
$$

mediante un producto vector-Jacobiano (VJP). Los grafos de diferenciación se descartan después de cada unidad de trabajo para evitar acumular memoria.

## Espectro mediante Lanczos y SLQ

La etapa A1 utiliza cuadratura estocástica de Lanczos. Para cada sonda de Rademacher normalizada $\mathbf q_1$, se construye una base de Krylov:

$$
\mathcal K_m(G,\mathbf q_1)
=\operatorname{span}
\{\mathbf q_1,G\mathbf q_1,\ldots,G^{m-1}\mathbf q_1\}.
$$

La recurrencia con reortogonalización completa produce una matriz tridiagonal $\mathbf T_m$. Sus autovalores aproximan posiciones espectrales y los cuadrados de la primera componente de sus autovectores proporcionan pesos de cuadratura.

El protocolo fijó:

- profundidad de Lanczos $m=64$;
- 64 sondas independientes;
- productos del decodificador en FP32;
- autodescomposición de $\mathbf T_m$ en FP64;
- referencia JVP por diferencia finita con paso 0.008;
- intervalos de dos lados del 99 % para resúmenes escalares.

Se estiman traza, energía espectral acumulada y dimensiones necesarias para explicar 95 % y 99 % de esa energía. Estas dimensiones describen la sensibilidad local del decodificador en puntos específicos; no son una estimación poblacional de todos los parches.

## Energía y longitud de un camino

Para camino $\gamma:[0,1]\rightarrow\mathcal Z$:

$$
E_D(\gamma)
=\int_0^1
\|J_D(\gamma(t))\dot\gamma(t)\|_2^2\,dt,
$$

$$
L_D(\gamma)
=\int_0^1
\|J_D(\gamma(t))\dot\gamma(t)\|_2\,dt.
$$

Con nudos $\mathbf z_0,\ldots,\mathbf z_K$ y $\Delta t=1/K$, la energía discreta usada para optimizar es

$$
\widehat E_D
=\sum_{j=0}^{K-1}
\frac{\|D(\mathbf z_{j+1})-D(\mathbf z_j)\|_2^2}{\Delta t}.
$$

Los extremos se fijan y solo se optimizan los $K-1$ nudos interiores. La inicialización es la línea recta latente. La optimización conserva el iterado de menor energía, no necesariamente el último.

## Puentes entre representantes

Para los ciclos exactos se comparan

$$
\mathbf e_k=E(R^k\mathbf x)
\quad\text{y}\quad
\mathbf a_k=\rho(r)^kE(\mathbf x).
$$

Que $D(\mathbf e_k)$ y $D(\mathbf a_k)$ sean parecidos no basta para afirmar que pertenecen a una misma fibra del decodificador. Se busca un camino conectado

$$
\beta_k:\mathbf e_k\rightarrow\mathbf a_k
$$

con baja energía y pequeño diámetro decodificado. Se reportan:

- error entre decodificaciones de los extremos;
- energía y longitud del puente;
- máxima separación de una decodificación intermedia respecto de los extremos;
- longitud latente;
- reversibilidad;
- sensibilidad al refinamiento temporal.

Un puente de bajo movimiento aporta evidencia de equivalencia aproximada. No prueba una fibra exacta. No encontrarlo tampoco prueba que no exista, porque la optimización es local.

## Cuatro lados codificados y prescritos

Se optimizan independientemente los cuatro lados de cada ciclo:

$$
\gamma_k:\mathbf z_k\rightarrow\mathbf z_{(k+1)\bmod4},
\qquad k=0,1,2,3.
$$

Cada lado usa $K=32$ segmentos y todos los grados de libertad del latente. No se proporciona una imagen de rotación intermedia como objetivo. Después se decodifican todos los nudos y se evalúan:

- energía y longitud;
- uniformidad de velocidad;
- error de inversión temporal;
- continuidad de extremos y tangentes;
- nitidez y deriva de contenido;
- sensibilidad al insertar puntos medios;
- covarianza de caminos bajo rotaciones exactas de $90^\circ$.

Los cuatro lados optimizados forman una curva cerrada por extremos, pero si sus tangentes no coinciden en las uniones se describe como camino cerrado por tramos, no como geodésica cerrada suave.

## Carta local inducida por un camino

Sea $V$ una base ortonormal de los desplazamientos observados a lo largo del primer lado. Se aplica el Jacobiano del decodificador a esas direcciones y se realiza una SVD del operador delgado. Los valores singulares se retienen cuando

$$
\sigma_i>32\,\epsilon_{\mathrm{FP32}}\,\sigma_{\max}.
$$

Esto define una base fija $U_0$ de direcciones visibles para el decodificador dentro del espacio observado por el camino. La carta afín es

$$
\phi(\boldsymbol\xi)=\mathbf z_0+U_0\boldsymbol\xi.
$$

Su métrica reducida es

$$
G_\phi(\boldsymbol\xi)
=[J_D(\phi(\boldsymbol\xi))U_0]^\mathsf T
[J_D(\phi(\boldsymbol\xi))U_0].
$$

Solo se usa lenguaje riemanniano intrínseco cuando este operador conserva rango completo y condicionamiento numérico sobre el recorrido. La carta es local y está determinada por un camino; no representa una acción compartida entre parches ni un cociente global.

## Continuación por disparo de una sola vez

El primer lado proporciona una velocidad inicial candidata. En coordenadas de carta:

$$
\dot{\boldsymbol\xi}=\mathbf c,
$$

$$
\dot{\mathbf c}
=-[J_D(\mathbf z)U_0]^\dagger
D^2D(\mathbf z)[U_0\mathbf c,U_0\mathbf c].
$$

La velocidad inicial se estima con una diferencia progresiva de segundo orden sobre el camino optimizado. Se integra con punto medio explícito RK2, ocho pasos por cuarto de vuelta, y se repite con 16 pasos como control temporal.

No se reajusta la velocidad para acertar el extremo. El primer cuarto es una prueba de consistencia de disparo; los tres cuartos siguientes se predicen sin consultar los anclajes futuros. Solo después de congelar la trayectoria se comparan

$$
\widehat{\mathbf z}_2,\widehat{\mathbf z}_3,
\widehat{\mathbf z}_4
$$

con los estados reservados.

## Transporte paralelo y defectos de retorno

En la carta fija, sea

$$
C_j=J_D(\mathbf z_j)U_0.
$$

Una aproximación discreta de transporte proyecta el vector decodificado entre espacios tangentes consecutivos:

$$
\mathbf c_{j+1}=C_{j+1}^\dagger C_j\mathbf c_j.
$$

Se registran por separado coordenadas, tangentes latentes y tangentes decodificadas. Los controles incluyen preservación de norma métrica, error hacia delante y atrás, y refinamiento del número de nudos.

Se distinguen:

- **defecto de cierre de punto:** diferencia entre posición final e inicial;
- **defecto de representante:** retorno a una decodificación parecida en otro vector latente;
- **retorno de tangente:** ángulo y discrepancia entre tangente transportada e inicial;
- **retorno de marco:** matriz que relaciona un marco transportado con el inicial.

Solo se usa el término holonomía cuando existe un lazo efectivamente cerrado y una identificación válida de espacios tangentes. Un retorno de punto no es holonomía y la holonomía no es torsión.

## Descomposición exacta de cuatro estados

Para cualquier cuádrupla $(\mathbf z_0,\mathbf z_1,\mathbf z_2,\mathbf z_3)$:

$$
\begin{aligned}
\mathbf c_0&=(\mathbf z_0+\mathbf z_1+\mathbf z_2+\mathbf z_3)/2,\\
\mathbf c_2&=(\mathbf z_0-\mathbf z_1+\mathbf z_2-\mathbf z_3)/2,\\
\mathbf c_c&=(\mathbf z_0-\mathbf z_2)/\sqrt2,\\
\mathbf c_s&=(\mathbf z_1-\mathbf z_3)/\sqrt2.
\end{aligned}
$$

Estos son sectores reales de la DFT de cuatro puntos. La identidad algebraica existe para cualquier cuádrupla; no demuestra una acción continua ni una variedad de baja dimensión. Se usa únicamente como diagnóstico secundario después de fijar los caminos principales.

## Estado del análisis en desarrollo

La etapa espectral A1 está cerrada. La etapa A2 calculó lados y puentes, pero todos los caminos relevantes continuaron reduciendo su energía al alcanzar el límite de optimización. Una continuación posterior también siguió mejorando en su último paso. Por ello, el siguiente trabajo no cambia el objetivo ni busca nuevos hiperparámetros: restaura los mismos caminos y el estado exacto del optimizador para continuar hasta que energía y gradiente se estabilicen.

Después, y solo en este orden, se planea:

1. interpolar caminos convergidos de $K=32$ a $K=64$ y reoptimizarlos para comprobar discretización y multiplicidad;
2. estudiar equivalencia del decodificador siguiendo modos pequeños del Jacobiano mediante predictor-corrector;
3. muestrear la órbita prescrita continua $\rho(\theta)\mathbf z_0$ y medir velocidad, densidad de energía y curvatura geodésica;
4. comparar caminos convergidos con disparo completo sin reducir el latente;
5. ensayar disparo múltiple periódico y cierre en una clase de equivalencia validada;
6. estudiar holonomía únicamente si el lazo, la equivalencia y la identificación tangente quedan definidos.

Estas etapas están en desarrollo. No deben describirse como resultados concluidos ni usarse para afirmar una estructura global del espacio latente.

# Análisis estadístico

Las comparaciones son pareadas: la misma WSI, parche, coordenada o presupuesto se evalúa con ambas representaciones.

## Reconstrucción

Cuando la cantidad principal se calcula sobre parches, la incertidumbre se obtiene mediante bootstrap por conglomerados de WSI. En cada réplica se seleccionan WSI con reemplazo y se transportan conjuntamente todos sus parches. Las mismas multiplicidades se aplican a ambos modelos.

Se utilizaron 10 000 réplicas y un intervalo percentil del 95 %. El tamaño inferencial corresponde a 23 WSI, no a 67 138 parches independientes.

## Diagnóstico

El bootstrap se estratifica por subtipo para conservar el soporte observado de CC, EC, HGSC, LGSC y MC. Se reportan macro-F1, exactitud balanceada, exactitud, entropía cruzada, F1 por clase y matrices de confusión.

## Tejido y eficiencia de etiquetas

El remuestreo se estratifica según las clases de tejido disponibles por WSI. La familia principal incluye las diferencias de macro-F1 en los cinco presupuestos y el área normalizada bajo la curva frente a $\log_{10}$ del número de etiquetas. Además de intervalos puntuales se usa una banda simultánea centrada para evitar leer cada presupuesto como una prueba independiente.

## Sondeos fijos

Los 25 parches y sus ángulos son un conjunto fijo de validación. Los ángulos son mediciones repetidas del mismo parche. Se reportan medianas, conteos pareados y, cuando corresponde, remuestreo por las 16 WSI representadas. No se presentan como una estimación poblacional de todas las WSI.

# Invariantes para una reimplementación

Una implementación metodológicamente equivalente debe mantener:

1. separación completa por WSI;
2. selección principal de tejido independiente de las máscaras;
3. máscaras negras interpretadas como desconocidas;
4. parches RGB de $256\times256$ y posterior $16\times32\times32$;
5. misma población, orden y objetivo para ambos VAE;
6. simetría objetivo continua $\mathrm{SO}(2)$, no un grupo discreto como sustituto;
7. campos ocultos $F_0/F_1$ y posterior escalar;
8. bases gaussianas-armónicas fijas y coeficientes aprendidos;
9. compuertas y normalizaciones compatibles con los tipos de campo;
10. salida RGB cruda sin `tanh`;
11. pérdida compartida y 60 000 actualizaciones;
12. congelación de los VAE antes de tareas supervisadas;
13. uso determinista de $\boldsymbol\mu$ en análisis posteriores;
14. selección de modelos únicamente con desarrollo y validación;
15. prueba sellada utilizada una sola vez y nunca para ajustar decisiones;
16. comparaciones pareadas sobre las mismas coordenadas;
17. WSI como unidad de incertidumbre para afirmaciones biológicas;
18. separación entre visualización descriptiva, control exacto y prueba geométrica;
19. distinción entre ciclo codificado y ciclo de acción prescrita;
20. prohibición de llamar fibra, cociente, geodésica u holonomía a un objeto que no cumpla las condiciones matemáticas correspondientes.

# Pseudocódigo del flujo completo

```python
# 1. Construir atlas
atlas = []
for wsi in sorted(selected_wsi):
    foreground = otsu_on_thumbnail_saturation(wsi.thumbnail)
    mask_fractions = project_sparse_mask_to_patch_grid(wsi.mask)
    for x, y in full_patch_grid(wsi.width, wsi.height, patch_size=256):
        fg = projected_foreground_fraction(foreground, x, y)
        fractions = mask_fractions[x, y]
        if fg > 0.60 or fractions.annotated > 0:
            atlas.append(record(wsi.id, x, y, fg, fractions))
atlas.sort(key=lambda row: (row.wsi_id, row.y, row.x))

# 2. Derivar vistas
vae_train, vae_validation = disjoint_wsi_patch_views(atlas_source)
foreground_view = [r for r in atlas if r.passes_otsu]
tissue_view = [
    r for r in atlas
    if r.annotated_fraction >= 0.10 and r.purity >= 0.95
]

# 3. Entrenar los dos VAE con el mismo protocolo
for model in (normal_vae, so2_vae):
    for step in range(60_000):
        corrupted, clean = denoising_pair(next_shared_patch(), rng_for(step))
        recon, mu, logvar = model(corrupted)
        loss = mae(recon, clean) + 0.1 * (1 - ssim(recon, clean))
        loss = loss + beta(step) * gaussian_kl(mu, logvar)
        optimizer_step(loss)
    freeze(model)

# 4. Extraer representaciones pareadas
for row, rgb_patch in stream_patches_in_atlas_order(atlas):
    normal_mu = normal_vae.encode(rgb_patch).mu.float()
    so2_mu = so2_vae.encode(rgb_patch).mu.float()
    write_paired_latent_records(row.identity, normal_mu, so2_mu)

# 5. Consumir vistas lógicas
evaluate_reconstruction(shared_test_patch_ids)
train_tissue_heads(nested_label_budgets)
train_wsi_mil_heads(complete_foreground_bags)
run_spatial_pca_and_rgb_probe(fixed_validation_patches)
run_rotation_and_decoded_action_probes(fixed_validation_patches)
run_functional_decoder_geometry(selected_fixed_patches)
```

# Qué puede modificarse en un experimento similar

Sin pretender reproducir exactamente este experimento, otro estudio puede cambiar órgano, tamaño de parche, grupos de simetría, anchos, presupuestos o clasificadores. Sin embargo, debe volver a justificar:

- cómo selecciona tejido sin usar las etiquetas que luego evaluará;
- cómo evita fuga de WSI;
- qué acciones geométricas son exactas en la cuadrícula y cuáles requieren interpolación;
- qué representación transforma cada campo;
- cómo conserva equivarianza en normalización y no linealidad;
- qué unidad estadística sostiene cada afirmación;
- qué información se usa para ajustar y cuál se reserva para prueba.

Cambiar un umbral, una forma latente o una convención de rotación no es un detalle menor si modifica la población o la hipótesis geométrica.

# Referencias esenciales

1. University of British Columbia. [UBC Ovarian Cancer Subtype Classification and Outlier Detection](https://www.kaggle.com/competitions/UBC-OCEAN/data), 2023.
2. N. Otsu. [A Threshold Selection Method from Gray-Level Histograms](https://doi.org/10.1109/TSMC.1979.4310076), 1979.
3. A. C. Ruifrok y D. A. Johnston. [Quantification of Histochemical Staining by Color Deconvolution](https://pubmed.ncbi.nlm.nih.gov/11531144/), 2001.
4. D. P. Kingma y M. Welling. [Auto-Encoding Variational Bayes](https://openreview.net/forum?id=33X9fd2-9FyZd), 2014.
5. K. He et al. [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html), 2016.
6. T. S. Cohen y M. Welling. [Steerable CNNs](https://openreview.net/forum?id=rJQKYt5ll), 2017.
7. M. Weiler y G. Cesa. [General E(2)-Equivariant Steerable CNNs](https://papers.nips.cc/paper/2019/hash/45d6637b718d0f24a237069fe41b0db4-Abstract.html), 2019.
8. Y. Wu y K. He. [Group Normalization](https://doi.org/10.1007/978-3-030-01261-8_1), 2018.
9. Z. Wang et al. [Image Quality Assessment: From Error Visibility to Structural Similarity](https://doi.org/10.1109/TIP.2003.819861), 2004.
10. I. Loshchilov y F. Hutter. [Decoupled Weight Decay Regularization](https://openreview.net/forum?id=Bkg6RiCqY7), 2019.
11. M. Ilse, J. Tomczak y M. Welling. [Attention-Based Deep Multiple Instance Learning](https://proceedings.mlr.press/v80/ilse18a.html), 2018.
12. J. Lee et al. [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html), 2019.
13. J. Ramapuram et al. [Theory, Analysis, and Best Practices for Sigmoid Self-Attention](https://openreview.net/forum?id=Zhdhg6n2OG), 2025.
14. T. Kouzelis et al. [EQ-VAE: Equivariance Regularized Latent Space for Improved Generative Image Modeling](https://proceedings.mlr.press/v267/kouzelis25a.html), 2025.
15. C. A. Field y A. H. Welsh. [Bootstrapping Clustered Data](https://doi.org/10.1111/j.1467-9868.2007.00593.x), 2007.
