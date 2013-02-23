package edu.sysu.shen.hadoop;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class WordsInCorpusTFIDF extends Configured implements Tool {

	private static final String OUTPUT_PATH = "/usr/shen/chinesewebkmeans/wordcount1";
	private static final String OUTPUT_PATH_2 = "/usr/shen/chinesewebkmeans/wordcount2";

	private static final String DICT_PATH = "/usr/shen/chinesewebkmeans/dict/dict.list";

	public static class WordsInCorpusTFIDFMapper extends
			Mapper<Text, Text, Text, Text> {

		private Text wordAndDoc = new Text();
		private Text wordAndCounters = new Text();
		private static Map<Long, Long> WEB_INDEX = new HashMap<Long, Long>();

		public void map(Text key, Text value, Context context)
				throws IOException, InterruptedException {
			String[] wordAndDoc = key.toString().split("@"); // 3/1500
			if (!WEB_INDEX.containsKey((Long.parseLong(wordAndDoc[1]))))
				WEB_INDEX.put(Long.parseLong(wordAndDoc[1]),
						Long.parseLong(wordAndDoc[1]));
			this.wordAndDoc.set(new Text(wordAndDoc[0]));
			this.wordAndCounters.set(wordAndDoc[1] + "=" + value.toString());
			context.write(this.wordAndDoc, this.wordAndCounters);
		}

	}

	public static class WordsInCorpusTFIDFReducer extends
			Reducer<Text, Text, Text, Text> {

		private static final DecimalFormat DF = new DecimalFormat(
				"###.########");
		private Text wordAtDocument = new Text();
		private Text tfidfCounts = new Text();
		private long numberOfDocumentsInCorpus;
		private static Map<String, Long> WORD_DICT = new HashMap<String, Long>();
		private static long WORDINDEX = 0;

		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			Configuration conf = context.getConfiguration();
			numberOfDocumentsInCorpus = conf.getLong("ALLDOCNUM", 1000000000);
			super.setup(context);
		}

		protected void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {

			if (!WORD_DICT.containsKey(key.toString())) {
				WORD_DICT.put(key.toString(), WORDINDEX);
				WORDINDEX++;
			}

			int numberOfDocumentsInCorpusWhereKeyAppears = 0;
			Map<String, String> tempFrequencies = new HashMap<String, String>();
			for (Text val : values) {
				String[] documentAndFrequencies = val.toString().split("=");
				if (Integer.parseInt(documentAndFrequencies[1].split("/")[0]) > 0) {
					numberOfDocumentsInCorpusWhereKeyAppears++;
				}
				tempFrequencies.put(documentAndFrequencies[0],
						documentAndFrequencies[1]);

			}
			for (String document : tempFrequencies.keySet()) {
				String[] wordFrequenceAndTotalWords = tempFrequencies.get(
						document).split("/");

				double tf = Double.valueOf(Double
						.valueOf(wordFrequenceAndTotalWords[0])
						/ Double.valueOf(wordFrequenceAndTotalWords[1]));

				double idf = Math
						.log10((double) numberOfDocumentsInCorpus
								/ (double) ((numberOfDocumentsInCorpusWhereKeyAppears == 0 ? 1
										: 0) + numberOfDocumentsInCorpusWhereKeyAppears));

				double tfIdf = tf * idf;
				this.wordAtDocument.set(key + "@" + document);
				this.tfidfCounts.set(DF.format(tfIdf));
				context.write(this.wordAtDocument, this.tfidfCounts);
			}
		}

		@Override
		protected void cleanup(Context context) throws IOException,
				InterruptedException {
			Configuration conf = context.getConfiguration();
			Path outPath = new Path(conf.get("DICTPATH"));
			FileSystem fs = FileSystem.get(conf);
			fs.delete(outPath, true);
			final SequenceFile.Writer out = SequenceFile.createWriter(fs, conf,
					outPath, Text.class, LongWritable.class);
			for (String word : WORD_DICT.keySet()) {
				out.append(new Text(word),
						new LongWritable(WORD_DICT.get(word)));
			}
			out.close();
			super.cleanup(context);
		}

	}

	public int run(String[] args) throws Exception {

		Configuration conf = getConf();
		FileSystem fs = FileSystem.get(conf);

		if (args[0] == null || args[1] == null) {
			System.out
					.println("You need to provide the arguments of the input and output");
		}

		Path userInputPath = new Path(args[0]);

		Path userOutputPath = new Path(args[1]);
		if (fs.exists(userOutputPath)) {
			fs.delete(userOutputPath, true);
		}

		Path wordFreqPath = new Path(OUTPUT_PATH);
		if (fs.exists(wordFreqPath)) {
			fs.delete(wordFreqPath, true);
		}

		Path wordCountsPath = new Path(OUTPUT_PATH_2);
		if (fs.exists(wordCountsPath)) {
			fs.delete(wordCountsPath, true);
		}

		Job job = new Job(conf, "Word Frequence In Document");
		job.setJarByClass(WordFrequenceInDocument.class);
		job.setMapperClass(WordFrequenceInDocument.WordFrequenceInDocMapper.class);
		job.setReducerClass(WordFrequenceInDocument.WordFrequenceInDocReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		SequenceFileInputFormat.addInputPath(job, userInputPath);
		SequenceFileOutputFormat.setOutputPath(job, new Path(OUTPUT_PATH));

		job.waitForCompletion(true);

		Configuration conf2 = getConf();
		Job job2 = new Job(conf2, "Words Counts");
		job2.setJarByClass(WordCountsInDocuments.class);
		job2.setMapperClass(WordCountsInDocuments.WordCountsForDocsMapper.class);
		job2.setReducerClass(WordCountsInDocuments.WordCountsForDocsReducer.class);
		job2.setOutputKeyClass(Text.class);
		job2.setOutputValueClass(Text.class);
		job2.setInputFormatClass(SequenceFileInputFormat.class);
		job2.setOutputFormatClass(SequenceFileOutputFormat.class);
		SequenceFileInputFormat.addInputPath(job2, new Path(OUTPUT_PATH));
		SequenceFileOutputFormat.setOutputPath(job2, new Path(OUTPUT_PATH_2));

		job2.waitForCompletion(true);

		Configuration conf3 = getConf();
		conf3.setInt("ALLDOCNUM", 1000000000);
		conf3.set("DICTPATH", DICT_PATH);
		Job job3 = new Job(conf3, "TF-IDF of Words in Corpus");
		job3.setJarByClass(WordsInCorpusTFIDF.class);
		job3.setMapperClass(WordsInCorpusTFIDFMapper.class);
		job3.setReducerClass(WordsInCorpusTFIDFReducer.class);
		job3.setOutputKeyClass(Text.class);
		job3.setOutputValueClass(Text.class);
		job3.setInputFormatClass(SequenceFileInputFormat.class);
		job3.setOutputFormatClass(SequenceFileOutputFormat.class);
		SequenceFileInputFormat.addInputPath(job3, new Path(OUTPUT_PATH_2));
		SequenceFileOutputFormat.setOutputPath(job3, userOutputPath);

		return job3.waitForCompletion(true) ? 0 : 1;

	}

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new WordsInCorpusTFIDF(),
				args);
		System.exit(res);
	}
}