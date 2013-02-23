package edu.sysu.shen.hadoop;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

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
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

/**
 * DocumentVetorBuid 根据tfidf建立文档向量 输出key:词语@文件 value:tfidf 输出key:文件编号
 * value:tfidf1:编号/tfidf2：编号/.../tfidfn：编号 使用hadoop1.0.4
 * 
 * @author 申国骏
 */
public class DocumentVetorBuid extends Configured implements Tool {

	// 输入目录
	private static final String INPUT_PATH = "/usr/shen/chinesewebkmeans/tfidf";
	// 输出目录
	private static final String OUTPUT_PATH = "/usr/shen/chinesewebkmeans/docvetor";
	private static final String DICT_PATH = "/usr/shen/chinesewebkmeans/dict/dict.list";
	private static final String CEN_PATH = "/usr/shen/chinesewebkmeans/center/centroid.list";

	public static class DocumentVetorMapper extends
			Mapper<Text, Text, LongWritable, Text> {

		// 记录文档编号
		private LongWritable docIndex = new LongWritable();
		// 记录词语以及tfidf
		private Text wordAndTfidf = new Text();
		private static Map<Long, Long> allDocNum = new HashMap<Long, Long>();

		@Override
		protected void cleanup(Context context) throws IOException,
				InterruptedException {
			Configuration conf = context.getConfiguration();
			Path cenPath = new Path(conf.get("CENTROIDPATH"));
			FileSystem fs = FileSystem.get(conf);
			fs.delete(cenPath, true);
			final SequenceFile.Writer out = SequenceFile.createWriter(fs,
					context.getConfiguration(), cenPath, LongWritable.class,
					LongWritable.class);
			Iterator<Entry<Long, Long>> iterator = allDocNum.entrySet()
					.iterator();
			while (iterator.hasNext()) {
				Map.Entry<Long, Long> entry = (Entry<Long, Long>) iterator
						.next();
				out.append(new LongWritable(entry.getKey()), new LongWritable(
						entry.getValue()));
			}
			out.close();
			super.cleanup(context);
		}

		/**
		 * @param context
		 *            输入 key:词语@文档编号 value:tfidf 输出 key:文档编号 value:文档中词语=tfidf
		 */
		public void map(Text key, Text value, Context context)
				throws IOException, InterruptedException {
			String[] wordAndDoc = key.toString().split("@");
			allDocNum.put(Long.parseLong(wordAndDoc[1]),
					Long.parseLong(wordAndDoc[1]));
			this.docIndex.set(Long.parseLong(wordAndDoc[1]));
			this.wordAndTfidf.set(wordAndDoc[0] + "=" + value.toString());
			context.write(docIndex, wordAndTfidf);
		}
	}

	public static class DocumentVetorReducer extends
			Reducer<LongWritable, Text, LongWritable, Text> {
		private Text tfIfd = new Text();
		// 记录词典中词语顺序
		private static Map<String, Long> WORD_DICT = new HashMap<String, Long>();
		private static List<Long> centers = new ArrayList<Long>();
		private static Map<Long, String> centerAndtifid = new HashMap<Long, String>();

		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			Configuration conf = context.getConfiguration();
			Path dictPath = new Path(conf.get("DICTPATH"));
			Path centorPath = new Path(conf.get("CENTROIDPATH"));
			int kvalue = conf.getInt("KVALUE", 10);
			List<Long> allCenters = new ArrayList<Long>();
			FileSystem fs = FileSystem.get(conf);

			SequenceFile.Reader reader = new SequenceFile.Reader(fs, dictPath,
					conf);
			Text key = new Text();
			LongWritable value = new LongWritable();
			while (reader.next(key, value)) {
				WORD_DICT.put(key.toString(), value.get());
			}
			reader.close();

			SequenceFile.Reader reader2 = new SequenceFile.Reader(fs,
					centorPath, conf);
			LongWritable key2 = new LongWritable();
			while (reader2.next(key2)) {
				allCenters.add(key2.get());
			}
			Collections.shuffle(allCenters);
			for (int i = 0; i < kvalue; i++) {
				centers.add(allCenters.get(i));
			}
			reader2.close();
			fs.delete(centorPath, true);
			super.setup(context);
		}

		/**
		 * @param context
		 *            输入 key:文档编号 value:<文档中词语1=tfidf1，文档中词语2=tfidf2...> 输出
		 *            key:文档编号 value:词典中对应顺序的tfidf1/tfidf2/.../tfidfn
		 */
		protected void reduce(LongWritable key, Iterable<Text> values,
				Context context) throws IOException, InterruptedException {
			StringBuilder allWordTfide = new StringBuilder();
			// 提取出在文档中出现的词语的tifidf并保存再文档向量的相应位置
			for (Text var : values) {
				String[] wordAndTfidf = var.toString().split("=");
				allWordTfide.append(WORD_DICT.get(wordAndTfidf[0]) + ":"
						+ wordAndTfidf[1] + "/");
			}
			// 将文档向量表示为tfidf1/tfidf2/.../tfidfn形式
			this.tfIfd.set(allWordTfide.toString());
			if (centers.contains(key.get())) {
				centerAndtifid.put(key.get(), allWordTfide.toString());
			}
			context.write(key, this.tfIfd);
		}

		@Override
		protected void cleanup(Context context) throws IOException,
				InterruptedException {
			Configuration conf = context.getConfiguration();
			FileSystem fs = FileSystem.get(conf);
			Path centorPath = new Path(conf.get("CENTROIDPATH"));
			fs.delete(centorPath, true);
			final SequenceFile.Writer out = SequenceFile.createWriter(fs, conf,
					centorPath, IntWritable.class, Text.class);
			Iterator<Entry<Long, String>> iterator = centerAndtifid.entrySet()
					.iterator();
			Text alltfidf = new Text();
			int i = 0;
			while (iterator.hasNext()) {
				Map.Entry<Long, String> entry = (Map.Entry<Long, String>) iterator
						.next();
				alltfidf.set(entry.getValue());
				out.append(new IntWritable(i), alltfidf);
				i++;
			}
			out.close();
			super.cleanup(context);
		}

	}

	public int run(String[] args) throws Exception {

		Configuration conf = getConf();
		FileSystem fs = FileSystem.get(conf);
		conf.set("CENTROIDPATH", CEN_PATH);
		conf.set("DICTPATH", DICT_PATH);
		conf.set("VECTORPATH", OUTPUT_PATH);
		conf.setInt("KVALUE", 20);
		Job job = new Job(conf, "Document Vetor Build");

		job.setJarByClass(DocumentVetorBuid.class);
		job.setMapperClass(DocumentVetorMapper.class);
		job.setReducerClass(DocumentVetorReducer.class);

		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(Text.class);

		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);

		if (fs.exists(new Path(OUTPUT_PATH))) {
			fs.delete(new Path(OUTPUT_PATH), true);
		}
		if (fs.exists(new Path(CEN_PATH))) {
			fs.delete(new Path(CEN_PATH), true);
		}
		FileInputFormat.setInputPaths(job, new Path(INPUT_PATH));
		FileOutputFormat.setOutputPath(job, new Path(OUTPUT_PATH));

		System.exit(job.waitForCompletion(true) ? 0 : 1);
		return 0;
	}

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new DocumentVetorBuid(),
				args);
		System.exit(res);
	}
}
