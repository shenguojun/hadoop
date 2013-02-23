package edu.sysu.shen.hadoop;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

/**
 * WordCountsInDocuments counts the total number of words in each document and
 * produces data with the relative and total number of words for each document.
 * 
 * @author Marcello de Sales (mdesales)
 */
public class WordCountsInDocuments extends Configured implements Tool {

	// where to put the data in hdfs when we're done
	private static final String OUTPUT_PATH = "/usr/shen/chinesewebkmeans/wordcount2";

	// where to read the data from.
	private static final String INPUT_PATH = "/usr/shen/chinesewebkmeans/wordcount";

	public static class WordCountsForDocsMapper extends
			Mapper<Text, IntWritable, Text, Text> {

		private Text docName = new Text();
		private Text wordAndCount = new Text();

		public WordCountsForDocsMapper() {
		}

		
		public void map(Text key, IntWritable value, Context context)
				throws IOException, InterruptedException {
			String wordAndDocCounter = value.toString();
			String[] wordAndDoc = key.toString().split("@");
			this.docName.set(wordAndDoc[1]);
			this.wordAndCount.set(wordAndDoc[0] + "=" + wordAndDocCounter);
			context.write(this.docName, this.wordAndCount);
		}
	}

	public static class WordCountsForDocsReducer extends
			Reducer<Text, Text, Text, Text> {

		private Text wordAtDoc = new Text();
		private Text wordAvar = new Text();

		protected void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			int sumOfWordsInDocument = 0;
			Map<String, Integer> tempCounter = new HashMap<String, Integer>();
			for (Text val : values) {
				String[] wordCounter = val.toString().split("=");
				tempCounter
						.put(wordCounter[0], Integer.valueOf(wordCounter[1]));
				sumOfWordsInDocument += Integer.parseInt(wordCounter[1]);
			}
			for (String wordKey : tempCounter.keySet()) {
				this.wordAtDoc.set(wordKey + "@" + key.toString());
				this.wordAvar.set(tempCounter.get(wordKey) + "/"
						+ sumOfWordsInDocument);
				context.write(this.wordAtDoc, this.wordAvar);
			}
		}
	}

	public int run(String[] args) throws Exception {

		Configuration conf = getConf();
		Job job = new Job(conf, "Words Counts");

		job.setJarByClass(WordCountsInDocuments.class);
		job.setMapperClass(WordCountsForDocsMapper.class);
		job.setReducerClass(WordCountsForDocsReducer.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);

		FileInputFormat.setInputPaths(job, new Path(INPUT_PATH));
		FileOutputFormat.setOutputPath(job, new Path(OUTPUT_PATH));

		return job.waitForCompletion(true) ? 0 : 1;
	}

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(),
				new WordCountsInDocuments(), args);
		System.exit(res);
	}
}
