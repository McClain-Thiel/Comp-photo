
def cm2in(num):
    return num*0.393701

def in2cm(num):
    return num/2.54

class_index = ['adult_male', 'adult_female', 'child_male', 'child_female']


adult_male_data = {"eye_dist": cm2in(6.4),
                   "height": 70,
                   "name": "Adult Male"
                   }

adult_female_data = {"eye_dist": cm2in(6.17),
                   "height": 63,
                     "name" : "Adult Female"
                   }


child_male_data = {"eye_dist": cm2in(4.9),
                   "height": 63,
                   "name": "Male Child"
                   }


child_female_data = {"eye_dist": cm2in(4.4),
                   "height": 59,
                     "name": "Female Child"
                   }


class_data = {"adult_male":adult_male_data,
              "adult_female":adult_female_data,
              "child_male":child_male_data,
              "child_female":child_female_data}
