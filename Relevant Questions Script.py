from flask import Flask
from flask import jsonify
from nltk.stem import WordNetLemmatizer 
from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.corpus import words
import nltk
import re
import numpy as np
from flask import request


from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

alphabets=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
model = SentenceTransformer('bert-base-nli-mean-tokens')
#nltk.download('wordnet')
#nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stop_words = list(get_stop_words('en'))         #About 900 stopwords
nltk_words = list(stopwords.words('english')) #About 150 stopwords
stop_words.extend(nltk_words)

jee_paper_2020=['If the magnetic field in a plane electromagnetic wave is given by ( ) 8 3 10 B x t T 3 10 sin 1.6 10 48 10 − =   +  , Then what will be expression for electric field?',
'The time period of revolution of electron in its ground state orbit in a hydrogen atom is 1.6 × 10–16. The frequency of revolution of the electron in its first excited state (in s–1 ) is',
'Consider a circular coil of wire carrying current I, forming a magnetic dipole. The magnetic flux through an infinite plane that contains the circular coil and excluding the circular coil area is given by ϕi. The magnetic flux through the area is given by ϕ0. Which of the following is correct ?',
'Visible light of wavelength 6000 × 10–8 cm falls normally on a single slit and produces a diffraction pattern. If it is found that the second diffraction minimum is at 60° from the central maximum. if the first minimum is produced at θ1, the θ1 is close to',
'The radius of gyration of a uniform rod if length l, about an axis passing through a point 1 2 away from the centre of the rod, and perpendicular to it',
'A 60 HP electric motor lifts an elevator having a maximum total load capacity of 2000 kg. If the frictional force on the elevator is 4000 N, the speed of the elevator at full load is close to: (1 HP = 746 W, g = 10 ms–2 )',
'Two infinite planes each with uniform surface charge density +σ are kept in such a way that the angle but them is 30°. The electric field in the region shown between them is given by',
'A satellite of mass m launched vertically upwards with an initial speed u from the surface of the earth. After it reaches height R (R = radius of the earth), it ejects a rocket of mass m/10 so that subsequently the satellite moves in a circular orbit. The kinetic energy of the rocket is (G is the gravitational constant; M is the mass of the earth)',
'A polarizer-analyser set is adjusted such that intensity of light coming out of the analyser is just 10% of the original intensity. Assuming that the polarizer. Analyser set does not absorb any light, the angle by which the analyser need to be rotated further to reduce the output intensity to be zero, is',
'A long solenoid of radius R carries a time (t) - dependent current I(t) = I0t (1-t). A ring of radius 2R is placed coaxially hear its middle. During the time interval 0 ≤ t ≤ 1, the induced current (IR) and the induced EMF (VR) in the ring changes as',
'Two moles of an ideal gas with 5 3 P V C C = are mixed with 3 moles of another ideal gas with 4 3 p v C C = . The value of p v C C for the mixture is',
'Speed of a transverse wave on a straight wire (mass 6.0 g, length 60 cm and area of cross-section 1.0 mm2 ) is 90 ms–1 . If the Youngs modules of wire is 16 × 1011 Nm–2 , the extension of wire over its natural length is',
'As shown in the figure, a bob a mass m is tied by a massless string whose other end portion is wound on a fly wheel (disc) of radius r and mass m. When released from rest the bob starts falling vertically. When it has covered a distance of h, the angular speed of the wheel will be',
'A LCR circuit behaves like a damped harmonic oscillator comparing it with a physical spring-mass damped oscillator having damping constant ‘b’, the correct equivalence would be',
'Three point particles of masses 1.0 k.g., 1.5 kg and 2.5 kg are placed at three corners of a right ∆ of sides 4.0 cm, 3.0 cm and 5.0 cm as shown. The centre of mass of the system is at a pt.',
'If we need a magnification of 375 from a compound microscope of tube length 150 mm and an objective of focal length 5m, the focal length of the eye-piece, should be close to',
'A litre of dry air at STP expands adiabatically to a volume of 3 litres. If γ = 1.40, the work done by air is: (31.4 = 4.6555) [Take air to be an ideal gas]',
'A parallel plate capacitor has plates of area A separated by distance d between them. It is filled with a dielectric which has a dielectric constant that varies as k(x) = K(1 + x) where x is the distance measured from one of the plates. If (d) << 1 , the total capacitance of the system is best given by the expression',
'A beam of electromagnetic radiation of intensity 6.4 × 10–5 w/cm2 is comprised of wavelength, λ = 310 nm. It falls normally on a metal (work function Φ = 2ev) of surface area 1 cm2 . If one in 103 photons ejects an electron, total number of electrons ejected in 1s is 10x (hc = 1240 eV nm, 1 eV = 1.6 × 10–19 J), then x is',
'A particle (m = 1 kg) slides down a frictionless track (AOC) starting from rest at a point A(height 2m). After reaching C, the particle continues to move freely in air as a projectile. When it reaching its highest point P (height 1m), the kinetic energy of the particle (in J) is : (Figure drawn is schematic and not to scale; take g = 10 ms–2 )',
'A loop ABCDEFA of straight edges has six corner points A (0, 0, 0), B(5, 0, 0), C(5, 5, 0), D(0, 5, 0), E(0, 5, 5) and F(0, 0, 5). The magnetic field in this region is ( ) ˆ ˆ B i k T = + 3 4 . The quantity of flux through the loop ABCDEFA(in Wb) is',
'A non-isotropic solid metal cube has coefficients of linear expansion as: 5 × 10–5 /°C along the x-axis and 5 × 10–6 /°C along the y and the z-axis. If coefficient of volume expansion of the solid C × 10–6 /°C then the value of C is',
'A Carnot engine operates between two reservoirs of temperature 900 K and 300 K. The engine performs 1200 J of work per cycle. The heat energy (in J) delivered by the engine to the low temperature reservoir, in a cycle, is',
]
question_paper_2020=["Write the mathematical form of Ampere Maxwell circuital law .","How does an increase in doping concentration affect the width of depletion layer of a p-n junction diode ?","A proton and an electron have equal speeds. Find the ratio of de Broglie  wavelengths associated with them .","The variation of the stopping potential ( Vo ) with the frequency ( v ) of the light incident on two different photosensitive surfaces M1 and M2 is shown in the figure . Identify the surface which has greater value of the  work function.","Explain the principle of working of a meter bridge. Draw the circuit diagram for determination of an unknown resistance using it","The space between the plates of a parallel plate capacitor is completely filled in two ways . In the first case , it is filled with a slab of dielectric  constant K . In the second case , it is filled with two slabs of equal  thickness and dielectric constants K1 and K2 respectively as shown in the  figure . The capacitance of the capacitor is same in the two cases. Obtain  the relationship between K, K1 and K2","Define the term ‘Half-life’ of a radioactive substance. Two different radioactive substances have half-lives T1 and T2 and number of  undecayed atoms at an instant, N1 and N2 , respectively . Find the ratio of their activities at that instant","Define wavefront of a travelling wave . Using Huygens principle, obtain the law of refraction at a plane interface when light passes from a denser to rarer medium .","Two long straight parallel wires A and B separated by a distance d, carry  equal current I flowing in same direction as shown in the figure","Find the magnetic field at a point P situated between them at a  distance x from one wire","Show graphically the variation of the magnetic field with distance  x for 0 < x < d.","Using Bohr’s atomic model , derive the expression for the radius of  nth orbit of the revolving electron in a hydrogen atom","Write two main observations of photoelectric effect experiment which could only be explained by Einstein’s photoelectric equation .","Draw a graph showing variation of photocurrent with the anode potential of a photocell","Explain the terms ‘depletion layer’ and ‘potential barrier’ in a  p-n junction diode . How are the (a) width of depletion layer , and  (b) value of potential barrier affected when the p-n junction is forward  biased ?","Two cells of emf E1 and E2 have their internal resistances r1 and r2 , respectively. Deduce an expression for the equivalent emf and internal resistance of their parallel combination when connected  across an external resistance R. Assume that the two cells are  supporting each other","In case the two cells are identical , each of emf E = 5 V and internal resistance r = 2 , calculate the voltage across the external resistance R = 10","Write an expression of magnetic moment associated with a current (I) carrying circular coil of radius r having N turns","Consider the above mentioned coil placed in YZ plane with its centre at the origin. Derive expression for the value of magnetic  field due to it at point (x, 0, 0)","Define current sensitivity of a galvanometer . Write its expression","A galvanometer has resistance G and shows full scale deflection for current Ig . (i) How can it be converted into an ammeter to measure current up to I0 (I0 > Ig) ? (ii) What is the effective resistance of this ammeter ?","A resistance R and a capacitor C are connected in series to a source V = V0  sin  Find  (a) The peak value of the voltage across the (i) resistance and  (ii) capacitor (b) The phase difference between the applied voltage and current .  Which of them is ahead ?","What is the effect on the interference fringes in Young’s double slit  experiment due to each of the following operations ? Justify your  answers (a) The screen is moved away from the plane of the slits (b) The separation between slits is increased (c) The source slit is moved closer to the plane of double slit","Write the expression for the speed of light in a material medium of  relative permittivity  and relative magnetic permeability .  (b) Write the wavelength range and name of the electromagnetic  waves which are used in (i) radar systems for aircraft navigation  and (ii) Earth satellites to observe the growth of the crops","The binding energies per nucleon of the parent nucleus , the daughter  nucleus and particle are 7·8 MeV , 7·835 MeV and 7·07 MeV respectively . Assuming the daughter nucleus to be formed in the unexcited state and neglecting its share in the energy of the reaction find the speed of the emitted particle (Mass of particle = 6·68  10–27 kg)","Draw circuit diagram and explain the working of a zener diode as a dc voltage regulator with the help of its I-V characteristic. (b) What is the purpose of heavy doping of p and n sides of a zener diode ?","Using Gauss law, derive expression for electric field due to a spherical shell of uniform charge distribution and radius R at a n point lying at a distance x from the centre of shell, such that","An electric field is uniform and acts along + x direction in the  region of positive x . It is also uniform with the same magnitude  but acts in - x direction in the region of negative x . The value of the field is E = 200 N/C for x > 0 and E = 200 N/C for x < 0 . A  right circular cylinder of length 20 cm and radius 5 cm has its  centre at the origin and its axis along the x-axis so that one flat face is at x = + 10 cm and the other is at x = – 10 cm","Find the expression for the potential energy of a system of two point charges q1 and q2 located at 1r and 2r, respectively in an external electric field E ","Draw equipotential surfaces due to an isolated point charge (– q) and depict the electric field lines","Three point charges + 1 C, – 1 C and + 2 C are initially infinite distance apart . Calculate the work done in assembling these charges at the vertices of an equilateral triangle of side 10 cm","An particle is accelerated through a potential difference of 10 kV  and moves along x-axis . It enters in a region of uniform magnetic field B = 2  10–3 T acting along y-axis Find the radius of its path (Take mass of -particle = 6·4  10–27 kg )","With the help of a labelled diagram, explain the working of a step-up transformer . Give reasons to explain the following (i) The core of the transformer is laminated (ii) Thick copper wire is used in windings . ","A conducting rod PQ of length 20 cm and resistance 0·1  rests on  two smooth parallel rails of negligible resistance AA and CC. It can slide on the rails and the arrangement is positioned between the poles of a permanent magnet producing uniform magnetic field B = 0·4 T. The rails , the rod and the magnetic field are in three mutually perpendicular directions as shown in the figure. If the ends A and C of the rails are short circuited , find the (i) external force required to move the rod with uniform velocity v = 10 cm/s, and (ii) power required to do so.","Draw the ray diagram of an astronomical telescope when the final image is formed at infinity. Write the expression for the resolving power of the telescope (b) An astronomical telescope has an objective lens of focal length  20 m and eyepiece of focal length 1 cm (i) Find the angular magnification of the telescope (ii) If this telescope is used to view the Moon , find the diameter of the image formed by the objective lens. Given the diameter of the Moon is 3·5  106 m and radius of lunar orbit is 108 m.","An object is placed in front of a concave mirror. It is observed that a virtual image is formed. Draw the ray diagram to show the image formation and hence derive the mirror equation . An object is placed 30 cm in front of a plano-convex lens with its spherical surface of radius of curvature 20 cm. If the refractive index of the material of the lens is 1·5, find the position and nature of the image formed."]


def clean_text(string: str, punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~''',stop_words=['the', 'a', 'and', 'is', 'be', 'will']) -> str:
    string = re.sub(r'https?://\S+|www\.\S+', '', string)

    # Cleaning the html elements
    string = re.sub(r'<.*?>', '', string)

    # Removing the punctuations
    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, "") 

    # Converting the text to lower
    string = string.lower()

    # Removing stop words
    string = ' '.join([word for word in string.split() if word not in stop_words])

    # Cleaning the whitespaces
    string = re.sub(r'\s+', ' ', string).strip()

    return(string) 

def cleaning_all_sentences(question_paper):
    final_question=[]
    for i in range(len(question_paper)):
        sent=clean_paragraphs(question_paper[i])
        sent = [w for w in sent.split() if not w in stop_words]
        sent=" ".join(sent)
        s=[word for word in sent.split() if word in words.words()]
        s=" ".join(s)
        final_question.append(s)
    return final_question

def main_model(model,corpus_embeddings,query,question_paper):
    final_clusters_list=[]
    model_k= KMeans(n_clusters=15, init='k-means++',random_state=44, max_iter=100, n_init=50)
    model_k.fit(corpus_embeddings)
    cluster_assignment = model_k.labels_
    for i in range(15):
        clusters_question=[]
        print()
        print(f'Cluster {i + 1} contains:')
        clust_sent = np.where(cluster_assignment == i)
        for k in clust_sent[0]:
            clusters_question.append(question_paper[k])
            print(f'- {question_paper[k]}')
        final_clusters_list.append(clusters_question)


    query_sent=clean_paragraphs(query)

    sent = [w for w in query_sent.split() if not w in stop_words]
    sent=" ".join(sent)
    sent=[lemmatizer.lemmatize(word) for word in sent.split()]
    sent=" ".join(sent)
    s=[word for word in sent.split() if word in words.words()]
    s=" ".join(s)
    y=model.encode(s)
    Y=y.reshape(1,-1)
    print("\n")
    print("----------------------------------------")
 
    print("Prediction of Query Cluster")
    print("QUERY :",s)

    prediction = model_k.predict(Y)
    print(prediction[0])
    
    return final_clusters_list[prediction[0]]
    print("Cluster",prediction+1)
    
    
def clean_paragraphs(paragraph):
    s=clean_text(paragraph)
    no_numeric=[word for word in s.split() if word.isalpha()]
    s=" ".join(no_numeric)
    words=[word for word in s.split() if word not in alphabets]
    s=" ".join(words)
    re_stop=[word for word in s.split() if word not in stopwords.words('english')]
    s=" ".join(re_stop)
    re_pre=[lemmatizer.lemmatize(word) for word in s.split() ]
    s=" ".join(re_pre)
    
    if s!=" ":
        if len(s.split())>1:
            return s
        else:
            return "none"
 






final_question_paper=cleaning_all_sentences(question_paper_2020)
corpus_embeddings=model.encode(final_question_paper)







app=Flask(__name__)

@app.route('/relevant_question', methods=['POST'])


def relevant_question():
    print(request)
    #request_data= request.data
    #print(request_data.decode('UTF-8'))
    request_data=request.get_json()
    
    query=request_data["query"]
    #request_data=request.json
    #data=jsonify(request_data)
    #print(data["query"])
    
    #query = request.args.get('query')
    #print(query)
    if request_data:
        if query:
        #if 'query' in request_data:
            #query=request_data['query']
            query_cleaned=clean_paragraphs(query)
            final_questions_cluster=main_model(model,corpus_embeddings,query_cleaned,question_paper_2020)
            result=[]
            for i in range(len(final_questions_cluster)):
                question=final_questions_cluster[i]
                result.append({'Query':query,
                    'question related':question,
                    'Question Paper':'CBSE BOARD QUESTION PAPER 2020'})
            final=tuple(i for i in result)
            return final
            
        else:
            return jsonify({"query":"null"})
    else:
            return jsonify({"query":"data not found"})




    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)



