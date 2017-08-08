package com.ml.weka.nn;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class WekaSupervisedLearning {
	
	public static void main(String[] args) throws Exception {
		
		System.out.println("Classification: ");
		
		MultilayerPerceptron classificationModel = readFromFile("C:\\models\\xor_classification.model");
		
		predict(classificationModel, true);
		
		//------------------------------------------
		System.out.println("Regression: ");
		
		MultilayerPerceptron regressionModel = readFromFile("C:\\models\\xor_regression.model");
		
		predict(regressionModel, false);
		
	}
	
	public static void predict(MultilayerPerceptron mlp, boolean classification) throws Exception {
		
		//initialization
		
		Instances predictionset = preparePredictionSet();
		
		//-------------------------------
		
		for(int i=0;i<predictionset.numInstances();i++){
			
			System.out.println((int) predictionset.instance(i).value(0)+" XOR "+(int) predictionset.instance(i).value(1));
			
			double[] distributions = mlp.distributionForInstance(predictionset.instance(i));
			
			double maxValue = -1; //define a small value initially
			double maxIndex = -1;
			
			for(int j=0;j<distributions.length;j++){
				
				System.out.println("class_"+j+": "+distributions[j]);
				
				if(distributions[j] > maxValue){
					
					maxIndex = j;
					maxValue = distributions[j];
					
				}
				
			}
			
			double classifiedIndex = mlp.classifyInstance(predictionset.instance(i));
			
			predictionset.instance(i).setClassValue(classifiedIndex);
			
			if(classification){
				
				String classifiedText = predictionset.instance(i).stringValue(predictionset.numAttributes() - 1);
				
				System.out.println("classified as: "+classifiedText+" ("+100*distributions[(int) maxIndex]+"%)");
				
			}
					
			System.out.println("--------------------------");
			
		}
		
		System.out.println();
		
	}
	
	public static Instances preparePredictionSet(){
		
		Instances predictionSet = initializeDataset();
		
		Instance item = null;
		
		item = new Instance(3); //x1, x2, result
		
		//0 XOR 0
		item.setValue(0, 0); //x1
		item.setValue(1, 0); //x2
		item.setValue(2, -1); //result, i do not know what the result is yet! initially assign -1.
		predictionSet.add(item);
		
		//------------------------
		
		//0 XOR 1
		item = new Instance(3); 
		item.setValue(0, 0); //x1
		item.setValue(1, 1); //x2
		item.setValue(2, -1); //result
		predictionSet.add(item);
		
		//------------------------
		
		//1 XOR 0
		item = new Instance(3); 
		item.setValue(0, 1); //x1
		item.setValue(1, 0); //x2
		item.setValue(2, -1); //result
		predictionSet.add(item);
		
		//------------------------
		
		//1 XOR 1
		item = new Instance(3); 
		item.setValue(0, 1); //x1
		item.setValue(1, 1); //x2
		item.setValue(2, -1); //result
		predictionSet.add(item);
		
		return predictionSet;
		
	}
	
	public static Instances initializeDataset(){
		
		FastVector fvWekaAttributes = new FastVector(3);
		
		Attribute x1 = new Attribute("x1");
		Attribute x2 = new Attribute("x2");
		Attribute result = new Attribute("result");
		
		fvWekaAttributes.addElement(x1);
		fvWekaAttributes.addElement(x2);
		fvWekaAttributes.addElement(result);
		
		Instances testset = new Instances("testset",fvWekaAttributes,0);
		
		return testset;
		
	}
	
	public static MultilayerPerceptron readFromFile(String dumpLocation) {
    	
    	MultilayerPerceptron mlp = new MultilayerPerceptron();
    	
    	//binary network is saved to following file
    	File file = new File(dumpLocation);
    	
    	FileInputStream fileInputStream = null;

    	//binary content will be stored in the binaryFile variable
    	byte[] binaryFile = new byte[(int) file.length()];
    	
    	try{
    		
    		fileInputStream = new FileInputStream(file);
    		fileInputStream.read(binaryFile);
    		fileInputStream.close();
    		
    	}
    	catch(Exception ex){
    		System.out.println(ex);
    	}
    	
    	try{
    		
        	mlp = (MultilayerPerceptron) deserialize(binaryFile);
        	
    	}
    	catch(Exception ex){
    		System.out.println(ex);
    	}
    	
    	return mlp;
    	
    }
    
    public static Object deserialize(byte[] bytes) throws Exception {
    	
    	ByteArrayInputStream b = new ByteArrayInputStream(bytes);
        
        ObjectInputStream o = new ObjectInputStream(b);
        
        return o.readObject();

    }

}
