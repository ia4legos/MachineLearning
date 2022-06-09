# DataPython

Ficheros de datos para análisis (extraidos de kaggle).

## Stroke Prediction Dataset 

Según la Organización Mundial de la Salud (OMS), el ictus es la segunda causa de muerte en el mundo, responsable de aproximadamente el 11% del total de fallecimientos. Este conjunto de datos se utiliza para predecir si un paciente tiene más o menos probabilidad de sufrir un ictus, en función de su género, edad, enfermedades y estatus de fumador. Cada fila en los datos proporciona información relevante sobre cada paciente. 

Attribute Information

1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not

*Note: "Unknown" in smoking_status means that the information is unavailable for this patient

[Descargar datos](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

## Rain in Australia

Este conjunto de datos contiene unos 10 años de observaciones meteorológicas diarias de muchos lugares de Australia. El objetivo es predecir si al día siguiente lloverá, hecho que viene reflejado en la variable RainTomorrow, con respuestas Sí (si la lluvia de ese día fue de 1mm o más) o No.

Attribute Information

1) Date: The date of observation
2) Location: The common name of the location of the weather station
3) MinTemp: The minimum temperature in degrees celsius
4) MaxTemp: The maximum temperature in degrees celsius
5) Rainfall: The amount of rainfall recorded for the day in mm
6) Evaporation: The so-called Class A pan evaporation (mm) in the 24 hours to 9am
7) Sunshine: The number of hours of bright sunshine in the day.
8) WindGustDir: The direction of the strongest wind gust in the 24 hours to midnight
9) WindGustSpeed: The speed (km/h) of the strongest wind gust in the 24 hours to midnight
10) WindDir9am: Direction of the wind at 3pm
11) WindSpeed9am: Wind speed (km/hr) averaged over 10 minutes prior to 9am
12) WindSpeed3pm: Wind speed (km/hr) averaged over 10 minutes prior to 3pm
13) Humidity9am: Humidity (percent) at 9am
14) Humidity3pm: Humidity (percent) at 3pm
15) Pressure9am: Atmospheric pressure (hpa) reduced to mean sea level at 9am
16) Pressure3pm: Atmospheric pressure (hpa) reduced to mean sea level at 3pm
17) Cloud9am: Fraction of sky obscured by cloud at 9am. This is measured in "oktas", which are a unit of eigths. It records how many
18) Cloud3pm: Fraction of sky obscured by cloud (in "oktas": eighths) at 3pm. See Cload9am for a description of the values
19) Temp9am: Temperature (degrees C) at 9am
20) Temp3pm: Temperature (degrees C) at 3pm
21) RainToday: Boolean: 1 if precipitation (mm) in the 24 hours to 9am exceeds 1mm, otherwise 0
22) RainTomorrow: The amount of next day rain in mm. Used to create response variable RainTomorrow. A kind of measure of the "risk".

[Descargar datos](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

Source & Acknowledgements

Observations were drawn from numerous weather stations. The daily observations are available from http://www.bom.gov.au/climate/data.
An example of latest weather observations in Canberra: http://www.bom.gov.au/climate/dwo/IDCJDW2801.latest.shtml

Definitions adapted from http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml
Data source: http://www.bom.gov.au/climate/dwo/ and http://www.bom.gov.au/climate/data.

Copyright Commonwealth of Australia 2010, Bureau of Meteorology.


## Pima Indian Diabetes

Este conjunto de datos procede del Instituto Nacional de Diabetes y Enfermedades Digestivas y Renales. El objetivo del estudio es predecir si un paciente tiene o no diabetes, basándose en determinadas mediciones diagnósticas incluidas en el conjunto de datos. Los datos considerados provienen de imponer una serie de restricciones a una base de datos más extensa; en concreto, todos los pacientes son mujeres de al menos 21 años de edad y de ascendencia india pima. Los datos contienen diversas variables médicas predictoras y una variable objetivo, Outcome, que indica si la paciente tiene o no diabetes. Las variables predictoras incluyen el número de embarazos que ha tenido la paciente, su IMC, su nivel de insulina, su edad, etc.

Attribute Information

1) Pregnancies: Number of times pregnant
2) Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3) BloodPressure: Diastolic blood pressure (mm Hg)
4) SkinThickness: Triceps skin fold thickness (mm)
5) Insulin: 2-Hour serum insulin (mu U/ml)
6) BMI: Body mass index (weight in kg/(height in m)^2)
7) DiabetesPedigreeF..: Diabetes pedigree function
8) Age: Age (years)
9) Outcome: Class variable (0 or 1) 268 of 768 are 1, the others are 0

Sources:
(a) Original owners: National Institute of Diabetes and Digestive and Kidney Diseases
(b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu)
Research Center, RMI Group Leader
Applied Physics Laboratory
The Johns Hopkins University
Johns Hopkins Road
Laurel, MD 20707
(301) 953-6231
(c) Date received: 9 May 1990

[Descarga datos](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)


## Breast Cancer Wisconsin

El conjunto de datos hace referencia a un estudio de cácer de mama a partir de las imágenes digitalizadas de un aspirado con aguja fina (FNA) de una masa mamaria. Se recogen las características del núcleo celular presente en la imagen. Esta base de datos está disponible en el UCI Machine Learning repository. [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

Este banco de datos puede ser encontrado en UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Attribute Information:

1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

## Mushroom classification

La "caza de setas" (también conocida como "shrooming" en inglés) está disfrutando de nuevos picos de popularidad. Conocer qué características significan una muerte segura y cuáles son más apetecibles es un aspecto muy importante en este proceso. Este conjunto de datos muestra las características de una muestra muy extensa de diferentes tipos de setas.

En concreto se incluyen descripciones de 23 especies de setas con agallas de la familia Agaricus y Lepiota, extraídas de The Audubon Society Field Guide to North American Mushrooms (1981). Cada especie se identifica como: "definitivamente comestible", "definitivamente venenosa", o "de comestibilidad desconocida y no recomendada". Esta última clase se combinó con la venenosa.

Attribute Information: (classes: edible=e, poisonous=p)

1) cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
2) cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
3) cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
4) bruises: bruises=t,no=f
5) odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
6) gill-attachment: attached=a,descending=d,free=f,notched=n
7) gill-spacing: close=c,crowded=w,distant=d
8) gill-size: broad=b,narrow=n
9) gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
10) stalk-shape: enlarging=e,tapering=t
11) stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
12) stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
13) stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
14) stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
15) stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
16) veil-type: partial=p,universal=u
17) veil-color: brown=n,orange=o,white=w,yellow=y
18) ring-number: none=n,one=o,two=t
19) ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
20) spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
21) population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
22) habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

[Descarga datos](https://www.kaggle.com/datasets/uciml/mushroom-classification)

## Hotel bookings

¿Se ha preguntado alguna vez cuál es la mejor época del año para reservar una habitación de hotel? ¿O la duración óptima de la estancia para conseguir la mejor tarifa diaria? ¿Y si quisiera predecir si un hotel tiene probabilidades de recibir un número desproporcionadamente alto de solicitudes especiales?

Este conjunto de datos de reservas de hoteles puede ayudarle a explorar esas cuestiones. Este conjunto de datos contiene información sobre las reservas de un hotel urbano y de un hotel turístico, e incluye información como la fecha en que se hizo la reserva, la duración de la estancia, el número de adultos, niños y/o bebés, y el número de plazas de aparcamiento disponibles, entre otras cosas.

Attribute Information:

1) hotel: (H1 = Resort Hotel or H2 = City Hotel)
2) is_canceled: Value indicating if the booking was canceled (1) or not (0)
3) lead_time: Number of days that elapsed between the entering date of the booking into the PMS and the arrival date
4) arrival_date_year: Year of arrival date
5) arrival_date_month: Month of arrival date
6) arrival_date_week_number: Week number of year for arrival date
7) arrival_date_day_of: Day of arrival date
8) stays_in_weekend_: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
9) stays_in_week_nigh: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel
10) adults: Number of adults
11) children: Number of children
12) babies: Number of babies
13) meal: Type of meal booked. Categories are presented in standard hospitality meal packages: Undefined/SC – no meal
14) country: Country of origin. Categories are represented in the ISO 3155–3:2013 format
15) market_segment: Market segment designation. In categories, the term “TA” means “Travel Agents” and “TO” means “Tour Operators”
16) distribution_channel: Booking distribution channel. The term “TA” means “Travel Agents” and “TO” means “Tour Operators”
17) is_repeated_guest: Number of previous bookings that were cancelled by the customer prior to the current booking
18) previous_bookings: Number of previous bookings not cancelled by the customer prior to the current booking
19) reserved_room_type: Code of room type reserved. Code is presented instead of designation for anonymity reasons.
20) assigned_room_type: Code for the type of room assigned to the booking. Sometimes the assigned room type differs from the reserved room type due
21) booking_changes: Number of changes/amendments made to the booking from the moment the booking was entered on the PMS
22) deposit_type: Indication on if the customer made a deposit to guarantee the booking. This variable can assume three categories: No
23) agent: ID of the travel agency that made the booking
24) company: ID of the company/entity that made the booking or responsible for paying the booking. ID is presented instead of designation for
25) days_in_waiting_list: Number of days the booking was in the waiting list before it was confirmed to the customer
26) customer_type: Type of booking, assuming one of four categories: Contract - when the booking has an allotment or other type of contract associated to
27) adr: Average Daily Rate as defined by dividing the sum of all lodging transactions by the total number of staying nights
28) required_car_parking_spaces: Number of car parking spaces required by the customer
29) total_of_special_requests: Number of special requests made by the customer (e.g. twin bed or high floor)
30) reservation_status: Reservation last status, assuming one of three categories: Canceled – booking was canceled by the customer; Check-Out
31) reservation_status_date: Date at which the last status was set. This variable can be used in conjunction with the ReservationStatus to

[Descargar datos](https://www.kaggle.com/datasets/abhi97/hotel-bookings)

## Student Alcohol Consumption

Los datos de consumo de alcohol entre adolescentes se obtuvieron en una encuesta realizada a los alumnos de los cursos de matemáticas y lengua portuguesa de enseñanza secundaria. Además de las variables refereidas al consumo se dipone de mucha información social, de género y de estudio interesante sobre los estudiantes. Este banco de datos se puede utilizar para un análisis exploratorio de datos o para intentar predecir la nota final de los estudiantes.

Attributes for student-mat.csv:

1) school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
2) sex - student's sex (binary: 'F' - female or 'M' - male)
3) age - student's age (numeric: from 15 to 22)
4) address - student's home address type (binary: 'U' - urban or 'R' - rural)
5) famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
6) Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
7) Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
8) Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
9) Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
10) Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
11) reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
12) guardian - student's guardian (nominal: 'mother', 'father' or 'other')
13) traveltime - home to school travel time (numeric: 1 - 1 hour)
14) studytime - weekly study time (numeric: 1 - 10 hours)
15) failures - number of past class failures (numeric: n if 1<=n<3, else 4)
16) schoolsup - extra educational support (binary: yes or no)
17) famsup - family educational support (binary: yes or no)
18) paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
19) activities - extra-curricular activities (binary: yes or no)
20) nursery - attended nursery school (binary: yes or no)
21) higher - wants to take higher education (binary: yes or no)
22) internet - Internet access at home (binary: yes or no)
23) romantic - with a romantic relationship (binary: yes or no)
24) famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
25) freetime - free time after school (numeric: from 1 - very low to 5 - very high)
26) goout - going out with friends (numeric: from 1 - very low to 5 - very high)
27) Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
28) Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
29) health - current health status (numeric: from 1 - very bad to 5 - very good)
30) absences - number of school absences (numeric: from 0 to 93)

These grades are related with the course subject, Math or Portuguese:

G1 - first period grade (numeric: from 0 to 20)
G2 - second period grade (numeric: from 0 to 20)
G3 - final grade (numeric: from 0 to 20, output target)


Source Information

P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.

Fabio Pagnotta, Hossain Mohammad Amran.
Email:fabio.pagnotta@studenti.unicam.it, mohammadamra.hossain '@' studenti.unicam.it
University Of Camerino

https://archive.ics.uci.edu/ml/datasets/STUDENT+ALCOHOL+CONSUMPTION

## California Housing Prices

Este es el conjunto de datos utilizado en el segundo capítulo del reciente libro de Aurélien Géron "Hands-On Machine learning with Scikit-Learn and TensorFlow". Sirve como una excelente introducción a la implementación de algoritmos de aprendizaje automático porque requiere una limpieza de datos rudimentaria, tiene una lista de variables fácilmente comprensible y se sitúa en un tamaño óptimo entre ser demasiado simple y demasiado engorroso.

Los datos contienen información del censo viviendas de California de 1990. Por lo tanto, aunque no pueda ayudar a predecir los precios actuales de la vivienda, proporciona un conjunto de datos introductorio accesible para enseñar a la gente los fundamentos del aprendizaje automático.

About this file

1) longitude: A measure of how far west a house is; a higher value is farther west
2) latitude: A measure of how far north a house is; a higher value is farther north
3) housingMedianAge: Median age of a house within a block; a lower number is a newer building
4) totalRooms: Total number of rooms within a block
5) totalBedrooms: Total number of bedrooms within a block
6) population: Total number of people residing within a block
7) households: Total number of households, a group of people residing within a home unit, for a block
8) medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
9) medianHouseValue: Median house value for households within a block (measured in US Dollars)
10) oceanProximity: Location of the house w.r.t ocean/sea

Acknowledgements
This data was initially featured in the following paper:
Pace, R. Kelley, and Ronald Barry. "Sparse spatial autoregressions." Statistics & Probability Letters 33.3 (1997): 291-297.

and I encountered it in 'Hands-On Machine learning with Scikit-Learn and TensorFlow' by Aurélien Géron.
Aurélien Géron wrote:
This dataset is a modified version of the California Housing dataset available from:
Luís Torgo's page (University of Porto)

[Descargar datos](https://www.kaggle.com/datasets/camnugent/california-housing-prices)


## Air pollution in Seoul

Este conjunto de datos contiene información sobre la medición de la contaminación atmosférica en Seúl (Corea del Sur). El Gobierno Metropolitano de Seúl proporciona muchos datos públicos, incluida la información sobre la contaminación atmosférica, a través de la "Plaza de Datos Abiertos".

En este subconjunto de datos se proporcionan los valores medios de seis contaminantes (SO2, NO2, CO, O3, PM10, PM2,5). Los datos se midieron cada seis horas entre 2017 y 2019. Los datos se midieron en 25 distritos de Seúl. Estos datos proceden del conjunto de datos de resumen de mediciones.

About this file

1) Measurament date
2) Station Code
3) Address
4) Latitude
5) Longitude
6) S02 (ppm): (Good) 0.02;	(Normal) 0.05;	(Bad) 0.15;	(Very bad) 1.0
7) NO2 (ppm): (Good) 0.03;	(Normal) 0.06;	(Bad) 0.2;	(Very bad) 2.0
8) O3 (ppm): (Good) 2.0;	(Normal) 9.0;	(Bad) 15.0;	(Very bad) 50.0
9) CO (ppm): (Good) 0.03;	(Normal) 0.09;	(Bad) 0.15;	(Very bad) 0.5
10) PM10 (Microgram/m3): (Good) 30.0;	(Normal) 80.0;	(Bad) 150.0;	(Very bad) 600.0
11) PM2.5 (Microgram/m3): (Good) 15.0;	(Normal) 35.0;	(Bad) 75.0;	(Very bad) 500.0

[Descargar datos](https://www.kaggle.com/datasets/bappekim/air-pollution-in-seoul)

## Cereals

Esta base de datos contiene información nutricional sobre distintos tipos de cereales.

Fields in the dataset:

1) Name: Name of cereal
2) mfr: Manufacturer of cereal (
  A = American Home Food Products;
  G = General Mills; 
  K = Kelloggs; 
  N = Nabisco; 
  P = Post; 
  Q = Quaker Oats; 
  R = Ralston Purina)
3) type: cold; hot
4) calories: calories per serving
5) protein: grams of protein
6) fat: grams of fat
7) sodium: milligrams of sodium
8) fiber: grams of dietary fiber
9) carbo: grams of complex carbohydrates
10) sugars: grams of sugars
11) potass: milligrams of potassium
12) vitamins: vitamins and minerals - 0, 25, or 100, indicating the typical percentage of FDA recommended
13) shelf: display shelf (1, 2, or 3, counting from the floor)
14) weight: weight in ounces of one serving
15) cups: number of cups in one serving
16) rating: a rating of the cereals (Possibly from Consumer Reports?)

Acknowledgements

These datasets have been gathered and cleaned up by Petra Isenberg, Pierre Dragicevic and Yvonne Jansen. The original source can be found here

[Descargar datos](https://www.kaggle.com/datasets/crawford/80-cereals)

## Winequality-red

Este conjunto de datos está relacionado con las variantes rojas del vino portugués "Vinho Verde". Para más detalles, consulte la referencia [Cortez et al., 2009]. Debido a cuestiones de privacidad y logística, sólo se dispone de variables fisicoquímicas (entradas) y sensoriales (la salida) (por ejemplo, no hay datos sobre tipos de uva, marca de vino, precio de venta del vino, etc.). Este conjunto de datos también está disponible en el repositorio de aprendizaje automático de la UCI, https://archive.ics.uci.edu/ml/datasets/wine+quality. Las variables provienen de tests psicoquímicos.


Input variables (based on physicochemical tests):

1) fixed acidity (most acids involved with wine or fixed or nonvolatile (do not evaporate readily))
2) volatile acidity (the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste)
3) citric acid (found in small quantities, citric acid can add 'freshness' and flavor to wines)
4) residual sugar (the amount of sugar remaining after fermentation stops, it's rare to find wines with less than 1 gram/liter and wines with greater than)
5) chlorides (the amount of salt in the wine)
6) free sulfur dioxide (the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents)
7) total sulfur dioxide (amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2)
8) density (the density of water is close to that of water depending on the percent alcohol and sugar content)
9) pH (describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the)
10) sulphates (a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and)
11) alcohol (the percent alcohol content of the wine)
12) quality (score between 0 and 10;  6.5 => "good")

[Descargar datos](https://www.kaggle.com/datasets/sh6147782/winequalityred)

## Melbourne Housing Snapshot (melb data set)

Melbourne real estate is BOOMING. Can you find the insight or predict the next big trend to become a real estate mogul… or even harder, to snap up a reasonably priced 2-bedroom unit? It was scraped from publicly available results posted every week from Domain.com.au. He cleaned it well, and now it's up to you to make data analysis magic. The dataset includes Address, Type of Real estate, Suburb, Method of Selling, Rooms, Price, Real Estate Agent, Date of Sale and distance from C.B.D.

Notes on Specific Variables

1) Rooms: Number of rooms
2) Price: Price in dollars
3) Method: S - property sold; SP - property sold prior; PI - property passed in; PN - sold prior not disclosed; SN - sold not disclosed; NB - no bid; VB - vendor bid; W - withdrawn prior to auction; SA - sold after auction; SS - sold after auction price not disclosed. N/A - price or highest bid not available.
4) Type: br - bedroom(s); h - house,cottage,villa, semi,terrace; u - unit, duplex; t - townhouse; dev site - development site; o res - other residential.
5) SellerG: Real Estate Agent
6) Date: Date sold
7) Distance: Distance from CBD
8) Regionname: General Region (West, North West, North, North east …etc)
9) Propertycount: Number of properties that exist in the suburb.
10) Bedroom2 : Scraped # of Bedrooms (from different source)
11) Bathroom: Number of Bathrooms
12) Car: Number of carspots
13) Landsize: Land Size
14) BuildingArea: Building Size
15) CouncilArea: Governing council for the area

[Descargar datos](https://www.kaggle.com/code/rhodamine6g/melbourne-housing-price-predictor)

## Marketing Analytics

The is a CSV file of 2240 observations (customers) with 28 variables related to marketing data. More specifically, the variables provide insights about:

1) ID: Customer's unique identifier
2) Year_Birth: Customer's birth year
3) Education: Customer's education level
4) Marital_Status: Customer's marital status
5) Income: Customer's yearly household income
6) Kidhome: Number of children in customer's household
7) Teenhome: Number of teenagers in customer's household
8) Dt_Customer: Date of customer's enrollment with the company
9) Recency: Number of days since customer's last purchase
10) MntWines: Amount spent on wine in the last 2 years
11) MntFruits: Amount spent on fruits in the last 2 years
12) MntMeatProducts: Amount spent on meat in the last 2 years
13) MntFishProducts: Amount spent on fish in the last 2 years
14) MntSweetProducts: Amount spent on sweets in the last 2 years
15) MntGoldProds: Amount spent on gold in the last 2 years
16) NumDealsPurchas: Number of purchases made with a discount
17) NumWebPurchases: Number of purchases made through the company's web site
18) NumCatalogPurcha: Number of purchases made using a catalogue
19) NumStorePurchase: Number of purchases made directly in stores
20) NumWebVisitsMont: Number of visits to company's web site in the last month
21) AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
22) AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
23) AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
24) AcceptedCmp1: 1 if customer accepted the offer in the 1nd campaign, 0 otherwise
25) AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
26) Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
27) Complain: 1 if customer complained in the last 2 years, 0 otherwise
28) Country: Customer's location

[Descargar datos](https://www.kaggle.com/code/chiamakauwaezuoke/marketing-data-analysis/data)

## Iris Flower Dataset

The Iris flower data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. The data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

The dataset contains a set of 150 records under 5 attributes - Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species).

Acknowledgements

This dataset is free and is publicly available at the UCI Machine Learning Repository

## Life expectancy

Context

Although there have been lot of studies undertaken in the past on factors affecting life expectancy considering demographic variables, income composition and mortality rates. It was found that affect of immunization and human development index was not taken into account in the past. Also, some of the past research was done considering multiple linear regression based on data set of one year for all the countries. Hence, this gives motivation to resolve both the factors stated previously by formulating a regression model based on mixed effects model and multiple linear regression while considering data from a period of 2000 to 2015 for all the countries. Important immunization like Hepatitis B, Polio and Diphtheria will also be considered. In a nutshell, this study will focus on immunization factors, mortality factors, economic factors, social factors and other health related factors as well. Since the observations this dataset are based on different countries, it will be easier for a country to determine the predicting factor which is contributing to lower value of life expectancy. This will help in suggesting a country which area should be given importance in order to efficiently improve the life expectancy of its population.

Content

The project relies on accuracy of data. The Global Health Observatory (GHO) data repository under World Health Organization (WHO) keeps track of the health status as well as many other related factors for all countries The data-sets are made available to public for the purpose of health data analysis. The data-set related to life expectancy, health factors for 193 countries has been collected from the same WHO data repository website and its corresponding economic data was collected from United Nation website. Among all categories of health-related factors only those critical factors were chosen which are more representative. It has been observed that in the past 15 years , there has been a huge development in health sector resulting in improvement of human mortality rates especially in the developing nations in comparison to the past 30 years. Therefore, in this project we have considered data from year 2000-2015 for 193 countries for further analysis. The individual data files have been merged together into a single data-set. On initial visual inspection of the data showed some missing values. As the data-sets were from WHO, we found no evident errors. Missing data was handled in R software by using Missmap command. The result indicated that most of the missing data was for population, Hepatitis B and GDP. The missing data were from less known countries like Vanuatu, Tonga, Togo, Cabo Verde etc. Finding all data for these countries was difficult and hence, it was decided that we exclude these countries from the final model data-set. The final merged file(final dataset) consists of 22 Columns and 2938 rows which meant 20 predicting variables. All predicting variables was then divided into several broad categories:​Immunization related factors, Mortality factors, Economical factors and Social factors.

Acknowledgements
The data was collected from WHO and United Nations website with the help of Deeksha Russell and Duan Wang.

Input variables

* Country: Country
* Year: Year
* Status: Developed or Developing status
* Life expectancy: Life Expectancy in age
* Adult Mortality: Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years per 1000 population)
* infant deaths: Number of Infant Deaths per 1000 population
* Alcohol: Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)
* percentage expenditure: Expenditure on health as a percentage of Gross Domestic Product per capita(%)
* Hepatitis B: Hepatitis B (HepB) immunization coverage among 1-year-olds (%)
* Measles: Measles - number of reported cases per 1000 population
* BMI: Average Body Mass Index of entire population
* under-five deaths: Number of under-five deaths per 1000 population
* Polio: Polio (Pol3) immunization coverage among 1-year-olds (%)
* Total expenditure: General government expenditure on health as a percentage of total government expenditure (%)
* Diphtheria: Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)
* HIV/AIDS: Deaths per 1 000 live births HIV/AIDS (0-4 years)
* GDP: Gross Domestic Product per capita (in USD)
* Population: Population of the country
* thinness 1-19 years: Prevalence of thinness among children and adolescents for Age 10 to 19 (% )
* thinness 5-9 years: Prevalence of thinness among children for Age 5 to 9(%)
* Income composition of resources: Human Development Index in terms of income composition of resources (index ranging from 0 to 1)
* Schooling: Number of years of Schooling(years)
