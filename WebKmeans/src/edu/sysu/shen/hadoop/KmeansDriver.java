package edu.sysu.shen.hadoop;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import edu.sysu.shen.hadoop.DocumentVetorBuid.DocumentVetorMapper;
import edu.sysu.shen.hadoop.DocumentVetorBuid.DocumentVetorReducer;
import edu.sysu.shen.hadoop.Kmeans.KmeansMapper;
import edu.sysu.shen.hadoop.Kmeans.KmeansReducer;
import edu.sysu.shen.hadoop.Kmeans.LastKmeansMapper;
import edu.sysu.shen.hadoop.Kmeans.LastKmeansReducer;
import edu.sysu.shen.hadoop.WordsInCorpusTFIDF.WordsInCorpusTFIDFMapper;
import edu.sysu.shen.hadoop.WordsInCorpusTFIDF.WordsInCorpusTFIDFReducer;

public class KmeansDriver {

	public static void main(String[] args) throws IOException,
			InterruptedException, ClassNotFoundException {

		// 计算tfidf过程
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		// 五个参数分别为[inputData path] [output path] [tmp path] [cluster number]
		// [maxIterations]
		if (args[0] == null || args[1] == null || args[2] == null
				|| args[3] == null || args[4] == null) {
			System.out
					.println("You need to provide the arguments of the input and output");

		}
		String inputDataPath = args[0];
		String outputPath = args[1];
		String tmpPath = args[2];
		int clusterNumber = Integer.parseInt(args[3]);
		int maxIterations = Integer.parseInt(args[4]);

		Path userInputPath = new Path(inputDataPath);

		Path wordFreqPath = new Path(tmpPath + "/wordcount1");
		if (fs.exists(wordFreqPath)) {
			fs.delete(wordFreqPath, true);
		}

		Path wordCountsPath = new Path(tmpPath + "/wordcount2");
		if (fs.exists(wordCountsPath)) {
			fs.delete(wordCountsPath, true);
		}

		Path tfidfPath = new Path(tmpPath + "/tfidf");
		if (fs.exists(tfidfPath)) {
			fs.delete(tfidfPath, true);
		}

		Path dictPath = new Path(tmpPath + "/dict/dict.list");
		if (fs.exists(dictPath)) {
			fs.delete(dictPath, true);
		}

		Job job = new Job(conf, "Calculate Word Frequence In Document");
		job.setJarByClass(WordFrequenceInDocument.class);
		job.setMapperClass(WordFrequenceInDocument.WordFrequenceInDocMapper.class);
		job.setReducerClass(WordFrequenceInDocument.WordFrequenceInDocReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		SequenceFileInputFormat.addInputPath(job, userInputPath);
		SequenceFileOutputFormat.setOutputPath(job, wordFreqPath);

		job.waitForCompletion(true);

		Configuration conf2 = new Configuration();
		Job job2 = new Job(conf2, "Words Counts In Document");
		job2.setJarByClass(WordCountsInDocuments.class);
		job2.setMapperClass(WordCountsInDocuments.WordCountsForDocsMapper.class);
		job2.setReducerClass(WordCountsInDocuments.WordCountsForDocsReducer.class);
		job2.setOutputKeyClass(Text.class);
		job2.setOutputValueClass(Text.class);
		job2.setInputFormatClass(SequenceFileInputFormat.class);
		job2.setOutputFormatClass(SequenceFileOutputFormat.class);
		SequenceFileInputFormat.addInputPath(job2, wordFreqPath);
		SequenceFileOutputFormat.setOutputPath(job2, wordCountsPath);

		job2.waitForCompletion(true);

		Configuration conf3 = new Configuration();
		conf3.setInt("ALLDOCNUM", 1000000000);
		conf3.set("DICTPATH", dictPath.toString());
		Job job3 = new Job(conf3, "Calculate TF-IDF of Words");
		job3.setJarByClass(WordsInCorpusTFIDF.class);
		job3.setMapperClass(WordsInCorpusTFIDFMapper.class);
		job3.setReducerClass(WordsInCorpusTFIDFReducer.class);
		job3.setOutputKeyClass(Text.class);
		job3.setOutputValueClass(Text.class);
		job3.setInputFormatClass(SequenceFileInputFormat.class);
		job3.setOutputFormatClass(SequenceFileOutputFormat.class);
		SequenceFileInputFormat.addInputPath(job3, wordCountsPath);
		SequenceFileOutputFormat.setOutputPath(job3, tfidfPath);

		job3.waitForCompletion(true);
		
		//建立文档向量/词表以及初始中心点
		Configuration conf4 = new Configuration();
		FileSystem fs4 = FileSystem.get(conf4);
		Path docVetorPath = new Path(tmpPath + "/docvetor");
		if (fs4.exists(docVetorPath)) {
			fs4.delete(docVetorPath, true);
		}
		
		Path centroidPath = new Path(tmpPath + "/centroid/centroid.list");
		if (fs4.exists(centroidPath)) {
			fs4.delete(centroidPath, true);
		}
		conf4.set("CENTROIDPATH", centroidPath.toString());
		conf4.set("DICTPATH", dictPath.toString());
		conf4.set("VECTORPATH", docVetorPath.toString());
		conf4.setInt("KVALUE", clusterNumber);
		Job job4 = new Job(conf4, "Build Document Vetor And Word Dict");

		job4.setJarByClass(DocumentVetorBuid.class);
		job4.setMapperClass(DocumentVetorMapper.class);
		job4.setReducerClass(DocumentVetorReducer.class);

		job4.setOutputKeyClass(LongWritable.class);
		job4.setOutputValueClass(Text.class);

		job4.setInputFormatClass(SequenceFileInputFormat.class);
		job4.setOutputFormatClass(SequenceFileOutputFormat.class);

		FileInputFormat.addInputPath(job4, tfidfPath);
		FileOutputFormat.setOutputPath(job4, docVetorPath);

		job4.waitForCompletion(true);
		
		
		//kmeans
		int iteration = 0;
		Configuration conf5 = new Configuration();
		conf5.set("num.iteration", iteration + "");
		conf5.set("DICTPATH", dictPath.toString());
		conf5.set("CENPATH", centroidPath.toString());
		Path out = new Path(tmpPath + "/clustering/depth_0");
		FileSystem fs5 = FileSystem.get(conf5);

		Job job5 = new Job(conf5);
		job5.setJobName("KMeansPrepare Clustering");

		job5.setMapperClass(KmeansMapper.class);
		job5.setReducerClass(KmeansReducer.class);
		job5.setJarByClass(Kmeans.class);

		FileInputFormat.addInputPath(job5, docVetorPath);
		
		if (fs5.exists(out))
			fs5.delete(out, true);

		if (fs5.exists(centroidPath))
			fs5.delete(out, true);

		if (fs5.exists(docVetorPath))
			fs5.delete(out, true);

		FileOutputFormat.setOutputPath(job5, out);
		job5.setInputFormatClass(SequenceFileInputFormat.class);
		job5.setOutputFormatClass(SequenceFileOutputFormat.class);

		job5.setOutputKeyClass(IntWritable.class);
		job5.setOutputValueClass(Text.class);

		job5.waitForCompletion(true);

		// 是否需要继续迭代
		long counter = maxIterations;
		// 迭代次数
		iteration++;
		while (counter > 0) {
			conf = new Configuration();
			conf.set("CENPATH", tmpPath + "/clustering/depth_"
					+ (iteration - 1) + "/" + "part-r-00000/");
			conf.set("num.iteration", iteration + "");
			conf.set("DICTPATH", dictPath.toString());
			job = new Job(conf);
			job.setJobName("KMeans Clustering " + iteration);

			job.setMapperClass(KmeansMapper.class);
			job.setReducerClass(KmeansReducer.class);
			job.setJarByClass(Kmeans.class);

			out = new Path(tmpPath + "/clustering/depth_" + iteration);

			FileInputFormat.addInputPath(job, docVetorPath);
			if (fs5.exists(out))
				fs5.delete(out, true);

			FileOutputFormat.setOutputPath(job, out);
			job.setInputFormatClass(SequenceFileInputFormat.class);
			job.setOutputFormatClass(SequenceFileOutputFormat.class);
			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(Text.class);

			job.waitForCompletion(true);
			counter--;
			//计算是否提前结束
			StringBuilder oldCentroid = new StringBuilder();
			StringBuilder newCentroid = new StringBuilder();
			Path oldCenPath = new Path(tmpPath + "/clustering/depth_"
					+ (iteration - 1) + "/part-r-00000");
			SequenceFile.Reader oldReader = new Reader(fs5, oldCenPath, conf);
			IntWritable key = new IntWritable();
			Text value = new Text();
			while (oldReader.next(key, value)) {
				oldCentroid.append(key.toString() + value.toString());
			}
			oldReader.close();
			Path newCenPath = new Path(tmpPath + "/clustering/depth_"
					+ (iteration) + "/part-r-00000");
			SequenceFile.Reader newReader = new Reader(fs5, newCenPath, conf);
			IntWritable key1 = new IntWritable();
			Text value1 = new Text();
			while (newReader.next(key1, value1)) {
				newCentroid.append(key1.toString() + value1.toString());
			}
			newReader.close();
			iteration++;
			if (newCentroid.toString().equals(oldCentroid.toString()))
				break;
		}

		conf = new Configuration();
		conf.set("CENPATH", tmpPath + "/clustering/depth_" + (iteration - 1)
				+ "/" + "part-r-00000/");
		conf.set("num.iteration", iteration + "");
		conf.set("DICTPATH", dictPath.toString());
		job = new Job(conf);
		job.setJobName("KMeans Last Clustering");

		job.setMapperClass(LastKmeansMapper.class);
		job.setReducerClass(LastKmeansReducer.class);
		job.setJarByClass(Kmeans.class);

		out = new Path(outputPath);
		if (fs.exists(out))
			fs.delete(out, true);

		FileInputFormat.addInputPath(job, docVetorPath);
		FileOutputFormat.setOutputPath(job, out);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(IntWritable.class);
		
		job.waitForCompletion(true);
		
		Path allTmpPath = new Path(tmpPath);
		fs5.delete(allTmpPath, true);

	}
}
