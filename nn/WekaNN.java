package com.ml.weka.nn;

/**
 * @author Sefik Ilkin Serengil
 * 
 * initialization: 2017-02-21
 * lastly updated: 2017-02-22
 * 
 */

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.math.BigDecimal;
import java.util.Date;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class WekaNN {
	
	public static void main(String[] args) {
		
		Date beginDate = new Date(); 
		
		//network variables			
		String backPropOptions = 
		"-L "+0.1 		//learning rate
                +" -M "+0 		//momentum
                +" -N "+10000000	//epoch. the larger epoch, the better results.
                +" -V "+0 		//validation
                +" -S "+0 		//seed
                +" -E "+0 		//error
                +" -H "+"3"; 		//hidden nodes. e.g. use "3,3" for 2 level hidden layer with 3 nodes
		
		String fileType = "arff"; //available file types: arff, txt
		String dumpLocation = "C:\\"; //specify where you want to store the binary network
		
		try{
						
			//prepare historical data
			Instances trainingset = retrieveHistoricalData(fileType);
			trainingset.setClassIndex(trainingset.numAttributes() - 1); //final attribute in a line stands for output
			
			//------------------------------
			//network training
			//deactivate this block to use already trained network
			MultilayerPerceptron mlp = new MultilayerPerceptron();
			mlp.setOptions(Utils.splitOptions(backPropOptions));
			mlp.buildClassifier(trainingset);
			
			//System.out.println("final weights:");
			//System.out.println(mlp);
			
			//store trained network
			byte[] binaryNetwork = serialize(mlp);
			writeToFile(binaryNetwork, dumpLocation);
			
			System.out.println("learning completed...");
			//------------------------------
			/*
			//activate this block to use already trained network
			MultilayerPerceptron mlp = readFromFile(dumpLocation);
			System.out.println("network weights and structure are load...");
			*/
			//------------------------------
			
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
			System.out.println("Mean Absolute Error: "+new BigDecimal(eval.meanAbsoluteError()));
			System.out.println("Root Mean Squared Error: "+eval.rootMeanSquaredError());
			System.out.println("Relative Absolute Error: "+eval.relativeAbsoluteError()+"%");
			System.out.println("Root Relative Squared Error: "+eval.rootRelativeSquaredError()+"%");
			System.out.println("Instances: "+eval.numInstances());
			
			Date endDate = new Date();
			
			System.out.println("\nprogram ends in "
					+(double)(endDate.getTime() - beginDate.getTime())/1000+" seconds\n");
			
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
	
    public static byte[] serialize(Object obj) throws Exception {
    	
        ByteArrayOutputStream b = new ByteArrayOutputStream();
        ObjectOutputStream o = new ObjectOutputStream(b);
        o.writeObject(obj);
        return b.toByteArray();
        
    }
    
    public static Object deserialize(byte[] bytes) throws Exception {
    	
    	ByteArrayInputStream b = new ByteArrayInputStream(bytes);
        
        ObjectInputStream o = new ObjectInputStream(b);
        
        return o.readObject();

    }
    
    public static void writeToFile(byte[] binaryNetwork, String dumpLocation) throws Exception {
    	  	
    	FileOutputStream stream = new FileOutputStream(dumpLocation+"trained_network.txt");
    	stream.write(binaryNetwork);
    	stream.close();
    	
    }
    
    public static MultilayerPerceptron readFromFile(String dumpLocation) {
    	
    	MultilayerPerceptron mlp = new MultilayerPerceptron();
    	
    	//binary network is saved to following file
    	File file = new File(dumpLocation+"trained_network.txt");
    	
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

}
