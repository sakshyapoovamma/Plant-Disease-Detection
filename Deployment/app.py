from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
#from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/my_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

#model._make_predict_function()      
print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    
    #update by ViPS
    img = cv2.imread(img_path)
    new_arr = cv2.resize(img,(100,100))
    new_arr = np.array(new_arr/255)
    new_arr = new_arr.reshape(-1, 100, 100, 3)
    

    
    preds = model.predict(new_arr)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads',f.filename )  #secure_filename(f.filename)
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        pred_class = preds.argmax()              # Simple argmax
 
        
        CATEGORIES = ['Pepper bell-Bacterial spot','Pepper bell-healthy',
            'Potato-Early blight' ,'Potato-Late blight', 'Potato-healthy',
            'Tomato-Bacterial spot' ,'Tomato-Early blight', 'Tomato-Late blight',
            'Tomato-Leaf Mold' ,'Tomato-Septoria leaf spot',
            'Tomato-Spider mites' ,'Tomato-Target Spot',
            'Tomato-YellowLeaf Curl Virus', 'Tomato-Mosaic virus',
            'Tomato-healthy']
        
        DEFINITION = ['Bacterial leaf spot, caused by Xanthomonas campestris pv. vesicatoria, is the most common and destructive disease for peppers in the eastern United States. It is a  gram-negative, rod-shaped bacterium that can survive in seeds and plant debris from one season to another (Frank et al. 2005). Different strains or races of the bacterium are cultivar-specific, causing disease symptoms in certain varieties due to stringent host specificity. Bacterial leaf spot can devastate a pepper crop by early defoliation of infected leaves and disfiguring fruit. In severe cases, plants may die as it is extremely difficult to find a cure once the disease takes hold. However, there are several options for growers to prevent it from occurring and spreading.\n\n-Using resistant varieties\n\n-Seed treatment\n\n-Foliage treatment\n-Integrated management\n\n'
                      ,'Your plant is healthy!',
                      'Early blight (EB) is a disease of potato caused by the fungus Alternaria solani. It is found wherever potatoes are grown. The disease primarily affects leaves and stems, but under favorable weather conditions, and if left uncontrolled, can result in considerable defoliation and enhance the chance for tuber infection. Premature defoliation may lead to considerable reduction in yield.The following measures will help prevent the occurrence of serious EB outbreaks, (1) Plant only diseasefree, certified seed. (2) Follow a complete and regular foliar fungicide spray program. (3) Practice good killing techniques to lessen tuber infections. (4) Allow tubers to mature before digging, dig when vines are dry, not wet, and avoid excessive wounding of potatoes during harvesting and handling. (5) Plow underall plant debris and volunteer potatoes after harvest. (6) Avoid replanting potatoes (and tomatoes or eggplants) in the affected fields for at least 2 years if severe outbreaks have been experienced. (7) Although no cultivar is immune to EB, several cultivars are moderately resistant and should be planted if blight is a continuing problem.',
                      'Late blight caused by the fungus Phytophthora infestans is the most important disease of potato that can result into crop failures in a short period if appropriate control measures are not adopted. Losses in potato yield can go as high as 80% in epidemic years.Here are methods to help control the disease:Destroy all cull and volunteer potatoes.Plant late blight-free seed tubers.Do not mix seed lots because cutting can transmit late blight.Use a seed piece fungicide treatment labeled for control of late blight. Recommended seed treatments include Revus, Reason and mancozeb.Avoid planting problem areas that may remain wet for extended periods or may be difficult to spray (the field near the center of the pivot, along powerlines and tree lines).Avoid excessive and/or nighttime irrigation.Eliminate sources of inoculum such as hairy nightshade weed species and volunteer potatoes.',
                      'Your plant is healthy!',
                      'Bacterial spot of tomato is a potentially devastating disease that, in severe cases, can lead to unmarketable fruit and even plant death.  Bacterial spot can occur wherever tomatoes are grown, but is found most frequently in warm, wet climates, as well as in greenhouses. Bacterial spot of tomato is caused by Xanthomonas vesicatoria, Xanthomonas euvesicatoria, Xanthomonas gardneri, and Xanthomonas perforans.Plant pathogen-free seed or transplants to prevent the introduction of bacterial spot pathogens on contaminated seed or seedlings.  If a clean seed source is not available or you suspect that your seed is contaminated, soak seeds in water at 122°F for 25 min. to kill the pathogens.  To keep leaves dry and to prevent the spread of the pathogens, avoid overhead watering (e.g., with a wand or sprinkler) of established plants and instead use a drip-tape or soaker-hose. ',
                      'Early blight of tomato is a serious disease requiring control measures, including fungicide applications. The disease occurs wherever tomato (and potato, Photos 6-8) is grown, and can cause severe defoliation, resulting in fewer, smaller fruit. Loss of yield is difficult to estimate, but probably at least 5%.Make sure that the seed is free from contamination of the fungus, by saving seed only from disease-free plants.Remove any volunteer tomato plants as well as weeds, especially those in the tomato family, from around nurseries, and in and around field plots.Space plants (60-90 cm) so that air circulates around them; this helps to dry the leaves rapidly after overhead irrigation or rain.Remove a few branches from the lower part of the plants to allow better airflow at the base.Prune any diseased leaves from the bottom of the plants as they become infected.',
                      'Tomato late blight is caused by the oomycete pathogen Phytophthora infestans (P. infestans). The pathogen is best known for causing the devastating Irish potato famine of the 1840s, which killed over a million people, and caused another million to leave the country.The first symptoms of late blight on tomato leaves are irregularly shaped, water-soaked lesions, often with a lighter halo or ring around them.These lesions are typically found on the younger, more succulent leaves in the top portion of the plant canopy.Plant early in the season to escape high disease pressure later in the season.Do not allow water to remain on leaves for long periods of time.Scout plants often and remove infected plants, infected fruit, volunteers and weeds.',
                      'Tomato leaf mold is typically only an issue in greenhouse and high-tunnel tomatoes.The disease is driven by high relative humidity (greater than 85%).Foliage is often the only part of the plant directly infected. Infection will cause infected leaves to wither and die, indirectly affecting yield.Use drip irrigation and avoid watering foliage.Space plants to provide good air movement between rows and individual plants.Stake, string or prune to increase airflow in and around the plant.Sterilize stakes, ties, trellises, etc. with 10% household bleach or commercial sanitizer.Circulate air in greenhouses or tunnels with vents and fans and by rolling up high tunnel sides to reduce humidity around plants.',
                      'Septoria leaf spot is caused by the fungus Septoria lycopersici. This fungus can attack tomatoes at any stage of development, but symptoms usually first appear on the older, lower leaves and stems when plants are setting fruit. Symptoms usually appear on leaves, but can occur on petioles, stems, and the calyx. The effects of Septoria leaf spot can be minimized by following a multifaceted approach to disease management that includes sanitary, cultural, and chemical methods. It is very important to eliminate initial sources of inoculum by removing or destroying as much of the tomato debris as possible after harvest in the fall. Alternatively, in large fields where plant removal is not practical, plant debris can be covered and buried by deep plowing. These simple sanitary practices can significantly reduce disease development the following year since they remove sources of the fungus that overwinter in the soil.',
                      'Tomato red spider mite, (Tetranychus evansi) is an established plant pest in NSW. It is a small, red coloured arachnid that feeds on the sap of plants.The vegetable crops of tomato, eggplant and potato are among the hosts of tomato red spider mite. Weeds in the Solanaceae family are also hosts of tomato red spider mite.The tomato red spider mite can be found on both sides of leaves but it prefers the undersides near the leaf veins. Feeding causes leaves to become yellowish white and mottled.Chemical resistance is a major problem overseas, requiring rotation of miticides.Manage weeds especially blackberry nightshade and glossy nightshade.Treat, remove or quarantine infested plants to prevent spread.Good farm hygiene practice - "Come Clean, Go Clean. "Ensure all staff and visitors are instructed in and adhere to your business management hygiene requirements.Source propagation material of a known high health status from reputable suppliers.',
                      'Target spot of tomato is caused by the fungal pathogen Corynespora cassiicola.1 The disease occurs on field-grown tomatoes in tropical and subtropical regions of the world. However, the disease also occurs on tomatoes grown in greenhouse and high tunnel production systems.The target spot fungus can infect all above-ground parts of the tomato plant. Plants are most susceptible as seedlings and just before and during fruiting. The initial foliar symptoms are pinpoint-sized, water-soaked spots on the upper leaf surface.Cultural practices for target spot management include improving airflow through the canopy by wider plant spacing and avoiding over-fertilizing with nitrogen, which can cause overly lush canopy formation. Pruning suckers and older leaves in the lower canopy can also increase airflow and reduce leaf wetness.Avoid planting tomatoes near old plantings. Inspect seedlings for target spot symptoms before transplanting. Manage weeds, which may serve as alternate hosts, and avoid the use of overhead irrigation.',
                      'Tomato yellow leaf curl virus is a species in the genus Begomovirus and family Geminiviridae. Tomato yellow leaf curl virus (TYLCV) infection induces severe symptoms on tomato plants and causes serious yield losses worldwide. TYLCV is persistently transmitted by the sweetpotato whitefly, Bemisia tabaci (Gennadius). Cultivars and hybrids with a single or few genes conferring resistance against TYLCV are often planted to mitigate TYLCV-induced losses. These resistant genotypes (cultivars or hybrids) are not immune to TYLCV.The management of TYLCV in tomato is difficult and expensive both in cultivation under a structure and open field production11. Many different approaches for controlling TYLCV disease such as removing whiteflies, killing intermediate weeds and changing cultivation season have been applied to decrease losses due to TYLCV because a single approach is not frequently effective and certain other approaches cannot be used in different agricultural environments and locations. Therefore a combination of chemical and biological control techniques for integrated pest management should be employed to reduce the population and migration of the whitefly vector and minimize or eliminate inoculum sources of TYLCV.',
                      'Tomato mosaic virus (ToMV) is a member of family tobamoviridae and belongs to the genus tobamovirus, which is a plant pathogenic virus. It is found worldwide and affects tomatoes and many other wide host range plants including many agricultural crops and weeds such as tobacco and beans, all of which can serve as inoculum sources. The tomato crop is highly susceptible to the Tomato mosaic virus (ToMV). The symptoms vary from tiles, wrinkle, reduction and curvature of leaflets, and irregular ripening of fruits. This disease requires attention because of its easy dissemination by contact, cultural practices, or contaminated seed.Treating mosaic virus is difficult and there are no chemicl controls like there are for fungal diseases. Tomato mosaic virus has been found to survive for up to 50 years in desiccated plant detritus! So tomato mosaic virus control then leans less on eliminating the disease and more on reducing and eliminating the virus sources and insect infestations. Control is mainly based on the use of virus-free seeds.',
                      'Your plant is healthy!']
        
        res=[]
        res2=DEFINITION[pred_class]
        res.append(CATEGORIES[pred_class])
        res.append(res2)

        return res


        
    return None


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
