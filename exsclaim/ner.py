from IPython.display import HTML
from promptify import OpenAI
from promptify import Prompter
import ast

api_key  = "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"
model = OpenAI(api_key) # or `HubModel()` for Huggingface-based inference
nlp_prompter = Prompter(model)

one_shot = "The application of graphene and its composites in oxygen reduction electrocatalysis: a perspective and review of recent progress The pressing necessity of a sustainable energy economy renders electrochemical energy conversion technologies, such as polymer electrolyte fuel cells or metal \u2013 air batteries, of paramount importance . The implementation of these technologies at scale still faces cost and operational durability challenges that stem from the conventionally used oxygen reduction reaction (ORR) electrocatalysts . While years of progress in ORR catalyst research has yielded some very attractive material designs, further advances are still required . Graphene entered the picture over 10 years ago, and scientists have only recently achieved a level of understanding regarding how its specific properties can be fine - tuned for electrocatalyst applications . This paper provides a critical review of the knowledge generated and progress realized over these past years for the development of graphene - based ORR catalysts . The first section discusses the application potential of graphene or modified graphene as platinum nanoparticle catalyst supports . The second section discusses the important role that graphene has played in the development of non-precious metal ORR catalysts, and more particularly its role in pyrolyzed transition metal \u2013 nitrogen \u2013 carbon complexes or as a support for inorganic nanoparticles . Finally the development of heteroatom doped graphene species is discussed, as this has been demonstrated as an excellent method to fine - tune the physicochemical properties and induce catalytic activity . Throughout this paper, clear differentiation is made between acidic and alkaline ORR catalysts, and some common misconceptions or improper testing practices used throughout the literature are revealed . Synthesis strategies and how they pertain to the resulting structure and electrochemical performance of graphene are discussed . In light of the large body of work done in this area, specific strategies are suggested for perpetuating the advancement of graphene - based ORR electrocatalysts . With concerted efforts it is one day likely that graphene - based catalysts will be a staple of electrochemical energy systems."
three_shot="Electrical properties of ceria - based oxides and their application to solid oxide fuel cells Ionic conductivities of ceria-alkaline-earth and -rare-earth oxide systems were investigated in relation to their structures, electrical conductivities, and reducibilities . Samaria and gadolinia - doped ceria samples exhibited the highest electrical conductivity in ceria - based oxides because of the close ionic radii of Sm3+ and Gd3+ to that of Ce4+ . The ionic conductivity of samaria- doped ceria was also measured by an ac four - probe method with electron blocking electrodes . A solid oxide fuel cell with a samaria ceria electrolyte produced high electric power, because of its highest oxygen ionic conductivity . The reduction of ceria electrolyte at the fuel side could be suppressed by a coating of stabilized zirconia thin film on the ceria surface . The anodic overvoltage of the doped ceria / anode interface was very small."
few_shot = [[one_shot, [ {"E": "electrochemical", "T": "applications"}, {"E": "catalysts", "T": "applications"}, {"E": "structure", "T": "material properties"}, {"E": "electrocatalyst", "T": "applications"}, {"E": "reduction", "T": "applications"}, {"E": "polymer", "T": "applications"}, {"E": "performance", "T": "material properties"}, {"E": "electrolyte", "T": "applications"}, {"E": "fuel", "T": "applications"}, {"E": "batteries", "T": "applications"}, {"E": "graphene", "T": "inorganic materials"}, {"E": "applications", "T": "applications"}, {"E": "durability", "T": "material properties"}, {"E": "platinum", "T": "inorganic materials"}, {"E": "activity", "T": "material properties"}, {"E": "reaction", "T": "applications"}, {"E": "catalyst","T": "applications"}, {"E": "electrocatalysts", "T": "applications"}, {"E": "nanoparticles", "T": "sample descriptors"}, {"E": "metal", "T": "applications"}, {"E": "composites", "T": "sample descriptors"}, {"E": "catalytic", "T": "material properties"}, {"E": "electrocatalysis", "T": "applications"}, {"E": "oxygen", "T": "applications"}, {"E": "conversion", "T": "applications"}, {"E": "carbon", "T": "inorganic materials"}, {"E": "doped", "T": "sample descriptors"}, {"E": "Graphene", "T": "inorganic materials"}, {"E": "air", "T": "applications"}, {"E": "(", "T": "applications"}, {"E": "systems", "T": "applications"}, {"E": "physicochemical", "T": "material properties"}, {"E": "energy", "T": "applications"}, {"E": "electrochemical", "T": "material properties"}, {"E": "technologies", "T": "applications"}, {"E": "cells", "T": "applications"}, {"E": "nanoparticle", "T": "sample descriptors"}, {"E": "properties", "T": "material properties"}, {"E": "ORR", "T": "applications"}, {"E": "\u2013", "T": "applications"}]],
            [three_shot, [{"E": "probe", "T": "characterization methods"}, {"E": "blocking", "T": "material applications"}, {"E": "ionic", "T": "material properties"}, {"E": "anode", "T": "material applications"}, {"E": "structures", "T": "material properties"}, {"E": "cell", "T": "material applications"}, {"E": "reducibilities", "T": "material properties"}, {"E": "surface", "T": "sample descriptors"}, {"E": "film", "T": "sample descriptors"}, {"E": "four", "T": "characterization methods"}, {"E": "oxide", "T": "inorganic materials"}, {"E": "electrolyte", "T": "material applications"}, {"E": "interface", "T": "sample descriptors"}, {"E": "overvoltage", "T": "material properties"}, {"E": "ceria", "T": "inorganic materials"}, {"E": "fuel", "T": "material applications"}, {"E": "ac", "T": "characterization methods"}, {"E": "solid", "T": "material applications"}, {"E": "samaria-", "T": "inorganic materials"}, {"E": "power", "T": "material properties"}, {"E": "oxides", "T": "inorganic materials"}, {"E": "Samaria", "T": "inorganic materials"}, {"E": "oxygen", "T": "material properties"}, {"E": "rare-earth", "T": "inorganic materials"}, {"E": "electric", "T": "material properties"}, {"E": "coating", "T": "material applications"}, {"E": "zirconia", "T": "inorganic materials"}, {"E": "conductivity", "T": "material properties"}, {"E": "electron", "T": "material applications"}, {"E": "electrical", "T": "material properties"}, {"E": "samaria", "T": "inorganic materials"}, {"E": "Electrical", "T": "material properties"}, {"E": "doped", "T": "sample descriptors"},{"E": "gadolinia", "T": "inorganic materials"}, {"E": "Ionic", "T": "material properties"}, {"E": "ceria-alkaline-earth", "T": "inorganic materials"}, {"E": "oxide", "T": "material applications"}, {"E": "cells", "T": "material applications"}, {"E": "conductivities", "T": "material properties"}, {"E": "electrodes", "T": "material applications"}, {"E": "thin", "T": "sample descriptors"}, {"E": "properties", "T": "material properties"}, {"E": "anodic", "T": "material properties"}, {"E": "method", "T": "characterization methods"}]]
            ]

def get_ner(caption):
  result = nlp_prompter.fit('ner.jinja',
                            domain      = 'solid state materials science',
                            text_input  = caption,
                            examples    = few_shot,
                            description = "Extract only entities related to solid state materials science and ignore all the other entities. Only one single word is allowed for each key as in the example. The entities should match exactly with the text.",
                            labels      = ["inorganic materials", "symmetry/phase labels", "sample descriptors", "material properties", "material applications", "synthesis methods", "characterization methods"] )
  data = ast.literal_eval(result['text'])

  # create an empty dictionary to hold the grouped values
  grouped_data = {"inorganic materials": [], "symmetry/phase labels": [], "sample descriptors": [], "material properties": [], "material applications": [], "synthesis methods": [], "characterization methods": []}
  labels = ["inorganic materials", "symmetry/phase labels", "sample descriptors", "material properties", "material applications", "synthesis methods", "characterization methods"]

  # loop over the dictionaries in the list
  for d in data[0]:
      # get the value of the 'T' key
      key = d['T']
      # get the value of the 'E' key
      value = d['E']
      # check if the key is already in the dictionary
      if key in grouped_data:
          # if it is, append the value to the existing list
          grouped_data[key].append(value)
      else:
          # if it's not, create a new list with the value
          grouped_data[key] = [value]

  # Add empty lists for the keys that are not in the data[0]
  for key in labels:
      if key not in grouped_data:
          grouped_data[key] = []

  text = caption
  grey_words = grouped_data['inorganic materials'] #["quick", "brown"]
  cyan_words = grouped_data['characterization methods']
  green_words = grouped_data['symmetry/phase labels']
  purple_words = grouped_data['sample descriptors']
  lightgreen_words = grouped_data['synthesis methods']
  yellow_words = grouped_data['material applications']
  mustard_words = grouped_data['material properties']

  # Highlight the words in the text with HTML tags and different colors
  for word in grey_words:
      text = text.replace(word, f"<span style='background-color: grey'>{word}</span>")

  for word in cyan_words:
      text = text.replace(word, f"<span style='background-color: cyan'>{word}</span>")

  for word in green_words:
      text = text.replace(word, f"<span style='background-color: #2C4E52'>{word}</span>")

  for word in purple_words:
      text = text.replace(word, f"<span style='background-color: magenta'>{word}</span>")

  for word in lightgreen_words:
      text = text.replace(word, f"<span style='background-color: lightgreen'>{word}</span>")

  for word in yellow_words:
      text = text.replace(word, f"<span style='background-color: yellow'>{word}</span>")

  for word in mustard_words:
      text = text.replace(word, f"<span style='background-color: #FFCC00'>{word}</span>")

  # Create an HTML string that displays the highlighted text
  html = f"<p>{text}</p>"

  # Display the HTML string in the notebook
  return html # display(HTML(html))
