package com.ml.weka.nn;

import java.io.BufferedReader;
import java.io.FileReader;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class WekaNN {
	
	public static void main(String[] args) {
		
		//network variables			
		String backPropOptions = 
				"-L "+0.1 		//learning rate
                +" -M "+0 		//momentum
                +" -N "+10000 	//epoch
                +" -V "+0 		//validation
                +" -S "+0 		//seed
                +" -E "+0 		//error
                +" -H "+"3"; 	//hidden nodes. e.g. use "3,3" for 2 level hidden layer with 3 nodes
		
		String fileType = "arff"; //available file types: arff, txt
		
		try{
						
			//prepare historical data
			Instances trainingset = retrieveHistoricalData(fileType);
			trainingset.setClassIndex(trainingset.numAttributes() - 1); //final attribute in a line stands for output
			
			//network training
			MultilayerPerceptron mlp = new MultilayerPerceptron();
			mlp.setOptions(Utils.splitOptions(backPropOptions));
			mlp.buildClassifier(trainingset);
			
			System.out.println("final weights:");
			System.out.println(mlp);
			
			//display actual and forecast values
			System.out.println("\nactual\tprediction");
			for(int i=0;i<trainingset.numInstances();i++){
				
				double actual = trainingset.instance(i).classValue();
				double prediction = mlp.distributionForInstance(trainingset.instance(i))[0];
				
				System.out.println(actual+"\t"+prediction);
				
			}
			
			//success metrics
			System.out.println("\nSuccess Metrics: ");
			Evaluation eval = new Evaluation(trainingset);
			eval.evaluateModel(mlp, trainingset);
			
			//display metrics
			System.out.println("Correlation: "+eval.correlationCoefficient());
			System.out.println("Mean Absolute Error: "+eval.meanAbsoluteError());
			System.out.println("Root Mean Squared Error: "+eval.rootMeanSquaredError());
			System.out.println("Relative Absolute Error: "+eval.relativeAbsoluteError()+"%");
			System.out.println("Root Relative Squared Error: "+eval.rootRelativeSquaredError()+"%");
			System.out.println("Instances: "+eval.numInstances());
			
		}
		catch(Exception ex){
			
			System.out.println(ex);
			
		}
		
	}
	
	public static Instances retrieveHistoricalData(String fileType){
		
		try{
			
			//read from weka format arff file
			if("arff".equals(fileType)){
				
				String historicalDataPath = System.getProperty("user.dir")+"\\dataset\\xor.arff";
				
				BufferedReader reader = new BufferedReader(new FileReader(historicalDataPath));
				Instances trainingset = new Instances(reader);
				reader.close();
				return trainingset;
				
			}
			
			//read from comma seperated txt file, transform it to weka format
			else if("txt".equals(fileType)){
				
				String historicalDataPath = System.getProperty("user.dir")+"\\dataset\\xor.txt";
				
				BufferedReader br = new BufferedReader(new FileReader(historicalDataPath));
				
				String line = br.readLine(); //header
				
				FastVector fvWekaAttributes = new FastVector(4);
				
				for(int i=0;i<line.split(",").length;i++){
					
					fvWekaAttributes.addElement(new Attribute(line.split(",")[i]));
					
				}
				
				Instances trainingset = new Instances("TrainingSet", fvWekaAttributes, 0); //initialize file
				
				while (line != null) {
					
					line = br.readLine();
					
					if(line != null){
						
						String[] items = line.split(",");
						
						Instance trainingItem = new Instance(items.length);
						
						int j = -1;
						
						for(int i=0;i<items.length;i++){
							
							trainingItem.setValue(++j, Double.parseDouble(items[i]));
							
						}
						
						trainingset.add(trainingItem);
						
					}
					
				}
				
				br.close();
				
				return trainingset;
			}
			
			
		}
		catch(Exception ex){
			
			System.out.println(ex);
			
		}
		
		return null;		
		
	}
	
}
